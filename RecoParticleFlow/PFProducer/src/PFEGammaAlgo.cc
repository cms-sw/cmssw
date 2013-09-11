#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFPhotonClusters.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFElectronExtraEqual.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <TFile.h>
#include <iomanip>
#include <algorithm>
#include <TMath.h>

// just for now do this
//#define PFLOW_DEBUG

#ifdef PFLOW_DEBUG
#define docast(x,y) dynamic_cast<x>(y)
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x)  edm::LogInfo(x)
#else
#define docast(x,y) reinterpret_cast<x>(y)
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

using namespace std;
using namespace reco;

namespace {
  typedef PFEGammaAlgo::PFSCElement SCElement;
  typedef PFEGammaAlgo::EEtoPSAssociation EEtoPSAssociation;
  typedef std::pair<CaloClusterPtr::key_type,CaloClusterPtr> EEtoPSElement;
  typedef PFEGammaAlgo::PFClusterElement ClusterElement;
  typedef PFEGammaAlgo::PFFlaggedElement PFFlaggedElement;
  typedef PFEGammaAlgo::PFSCFlaggedElement SCFlaggedElement;
  typedef PFEGammaAlgo::PFKFFlaggedElement KFFlaggedElement;
  typedef PFEGammaAlgo::PFGSFFlaggedElement GSFFlaggedElement;
  typedef PFEGammaAlgo::PFClusterFlaggedElement ClusterFlaggedElement;

  typedef std::unary_function<const ClusterFlaggedElement&,
			      bool> ClusterMatcher;  
  
  typedef std::unary_function<const PFFlaggedElement&,
			      bool> PFFlaggedElementMatcher; 
  typedef std::binary_function<const PFFlaggedElement&,
			       const PFFlaggedElement&,
			       bool> PFFlaggedElementSorter; 
  
  typedef std::unary_function<const reco::PFBlockElement&,
			      bool> PFElementMatcher; 

  typedef std::unary_function<const PFEGammaAlgo::ProtoEGObject&,
			      bool> POMatcher; 
  
  typedef std::unary_function<PFFlaggedElement&, 
			      ClusterFlaggedElement> ClusterElementConverter;

  struct SumPSEnergy : public std::binary_function<double,
						 const ClusterFlaggedElement&,
						 double> {
    reco::PFBlockElement::Type _thetype;
    SumPSEnergy(reco::PFBlockElement::Type type) : _thetype(type) {}
    double operator()(double a,
		      const ClusterFlaggedElement& b) {

      return a + (_thetype == b.first->type())*b.first->clusterRef()->energy();
    }
  };

  bool comparePSMapByKey(const EEtoPSElement& a,
			 const EEtoPSElement& b) {
    return a.first < b.first;
  }
  
  struct UsableElementToPSCluster : public ClusterElementConverter {
    ClusterFlaggedElement operator () (PFFlaggedElement& elem) {      
      const ClusterElement* pselemascluster = 
	docast(const ClusterElement*,elem.first);  
      if( reco::PFBlockElement::PS1 != pselemascluster->type() &&
	  reco::PFBlockElement::PS2 != pselemascluster->type()    ) {
	std::stringstream ps_err;
	pselemascluster->Dump(ps_err,"\t");
	throw cms::Exception("UseableElementToPSCluster()")
	  << "This element is not a PS cluster!" << std::endl
	  << ps_err.str() << std::endl;
      }
      if( elem.second == false ) {
	std::stringstream ps_err;
	pselemascluster->Dump(ps_err,"\t");
	throw cms::Exception("UsableElementToPSCluster()")
	  << "PS Cluster matched to EE is already used! "
	  << "This should be impossible!" << std::endl
	  << ps_err.str() << std::endl;
      }
      elem.second = false; // flag as used!   
      return std::make_pair(pselemascluster,true);         
    }
  };


  // (soon-to-be-non)trivial examples of matching and cleaning kernel functions
  struct KFMatchesToCluster : public ClusterMatcher {
    const KFFlaggedElement* track;
    bool operator() (const ClusterFlaggedElement&) {
      return false;
    }
  };
  
  struct GSFMatchesToCluster : public ClusterMatcher {
    const GSFFlaggedElement* track;
    bool operator() (const ClusterFlaggedElement&) {
      return false;
    }
  };
  
  struct SeedMatchesToSCElement : public PFFlaggedElementMatcher {
    reco::SuperClusterRef _scfromseed;
    SeedMatchesToSCElement(const reco::ElectronSeedRef& s) {
      _scfromseed = s->caloCluster().castTo<reco::SuperClusterRef>();
    }
    bool operator() (const PFFlaggedElement& elem) {
      const SCElement* scelem = docast(const SCElement*,elem.first);
      if( _scfromseed.isNull() || !elem.second || !scelem) return false;     
      return ( _scfromseed->seed()->seed() == 
	       scelem->superClusterRef()->seed()->seed() );
    }
  };

  struct SCSubClusterMatchesToElement : public PFFlaggedElementMatcher {
    const reco::CaloCluster_iterator begin, end;
    SCSubClusterMatchesToElement(const reco::CaloCluster_iterator& b,
				 const reco::CaloCluster_iterator& e) : 
      begin(b),
      end(e) { }
    bool operator() (const PFFlaggedElement& elem) {
      const ClusterElement* cluselem = 
	docast(const ClusterElement*,elem.first);
      if( !elem.second || !cluselem) return false;
      reco::CaloCluster_iterator cl = begin;
      for( ; cl != end; ++cl ) {
	if((*cl)->seed() == cluselem->clusterRef()->seed()) {
	  return true;
	}
      }
      return false;
    }
  }; 

  struct SeedMatchesToProtoObject : public POMatcher {
    reco::SuperClusterRef _scfromseed;
    SeedMatchesToProtoObject(const reco::ElectronSeedRef& s) {
      _scfromseed = s->caloCluster().castTo<reco::SuperClusterRef>();
    }
    bool operator() (const PFEGammaAlgo::ProtoEGObject& po) {
      if( _scfromseed.isNull() || !po.parentSC ) return false;      
      return ( _scfromseed->seed()->seed() == 
	       po.parentSC->superClusterRef()->seed()->seed() );
    }
  }; 
  
  template<bool useConvs=false>
  bool elementNotCloserToOther(const reco::PFBlockRef& block,
			       const PFBlockElement::Type& keytype,
			       const size_t key, 
			       const PFBlockElement::Type& valtype,
			       const size_t test,
			       const float EoPin_cut = 1.0e6) {
    constexpr reco::PFBlockElement::TrackType ConvType =
	reco::PFBlockElement::T_FROM_GAMMACONV;
    // this is inside out but I just want something that works right now
    switch( keytype ) {
    case reco::PFBlockElement::GSF:
      {
	const reco::PFBlockElementGsfTrack* elemasgsf  =  
	  docast(const reco::PFBlockElementGsfTrack*,
		 &(block->elements()[key]));
	if( elemasgsf && valtype == PFBlockElement::ECAL ) {
	  const ClusterElement* elemasclus =
	 reinterpret_cast<const ClusterElement*>(&(block->elements()[test]));
	  float cluster_e = elemasclus->clusterRef()->energy();
	  float trk_pin   = elemasgsf->Pin().P();
	  if( cluster_e / trk_pin > EoPin_cut ) return false;
	}
      }
      break;
    case reco::PFBlockElement::TRACK:
      {
	const reco::PFBlockElementTrack* elemaskf  = 
	  docast(const reco::PFBlockElementTrack*,
		 &(block->elements()[key]));
	if( elemaskf && valtype == PFBlockElement::ECAL ) {
	  const ClusterElement* elemasclus =
	  reinterpret_cast<const ClusterElement*>(&(block->elements()[test]));
	  float cluster_e = elemasclus->clusterRef()->energy();
	  float trk_pin   = 
	    std::sqrt(elemaskf->trackRef()->innerMomentum().mag2());
	  if( cluster_e / trk_pin > EoPin_cut ) return false;
	}
      }	
      break;
    default:
      break;
    }	        

    const double dist = 
      block->dist(key,test,block->linkData(),reco::PFBlock::LINKTEST_ALL);
    if( dist == -1.0 ) return false; // don't associate non-linked elems
    std::multimap<double, unsigned> dists_to_val; 
    block->associatedElements(test,block->linkData(),dists_to_val,keytype,
			      reco::PFBlock::LINKTEST_ALL);   
    for( const auto& valdist : dists_to_val ) {
      const size_t idx = valdist.second;
      // check track types for conversion info
      switch( keytype ) {
      case reco::PFBlockElement::GSF:
	{
	  const reco::PFBlockElementGsfTrack* elemasgsf  =  
	    docast(const reco::PFBlockElementGsfTrack*,
		   &(block->elements()[idx]));
	  if( !useConvs && elemasgsf->trackType(ConvType) ) return false;
	  if( elemasgsf && valtype == PFBlockElement::ECAL ) {
	    const ClusterElement* elemasclus =
	      docast(const ClusterElement*,&(block->elements()[test]));
	    float cluster_e = elemasclus->clusterRef()->energy();
	    float trk_pin   = elemasgsf->Pin().P();
	    if( cluster_e / trk_pin > EoPin_cut ) continue;
	  }
	}
	break;
      case reco::PFBlockElement::TRACK:
	{
	  const reco::PFBlockElementTrack* elemaskf  = 
	    docast(const reco::PFBlockElementTrack*,
		   &(block->elements()[idx]));
	  if( !useConvs && elemaskf->trackType(ConvType) ) return false;
	  if( elemaskf && valtype == PFBlockElement::ECAL ) {
	    const ClusterElement* elemasclus =
	    reinterpret_cast<const ClusterElement*>(&(block->elements()[test]));
	    float cluster_e = elemasclus->clusterRef()->energy();
	    float trk_pin   = 
	      std::sqrt(elemaskf->trackRef()->innerMomentum().mag2());
	    if( cluster_e / trk_pin > EoPin_cut ) continue;
	  }
	}	
	break;
      default:
	break;
      }	        
      if( valdist.first < dist && idx != key ) {
	return false; // false if closer element of specified type found
      }
    }
    return true;
  }

  struct CompatibleEoPOut : public PFFlaggedElementMatcher {
    const reco::PFBlockElementGsfTrack* comp;
    CompatibleEoPOut(const reco::PFBlockElementGsfTrack* c) : comp(c) {}
    bool operator()(const PFFlaggedElement& e) const {
      if( PFBlockElement::ECAL != e.first->type() ) return false;
      const ClusterElement* elemascluster = 
	docast(const ClusterElement*,e.first);
      const float gsf_eta_diff = std::abs(comp->positionAtECALEntrance().eta()-
					  comp->Pout().eta());
      const reco::PFClusterRef cRef = elemascluster->clusterRef();
      return ( gsf_eta_diff <= 0.3 && cRef->energy()/comp->Pout().t() <= 5 );
    }
  };

  template<class TrackElementType>
  struct IsConversionTrack : PFFlaggedElementMatcher {
    bool operator()(const PFFlaggedElement& e) {
      constexpr reco::PFBlockElement::TrackType ConvType =
	reco::PFBlockElement::T_FROM_GAMMACONV;
      const TrackElementType* elemastrk = 
	docast(const TrackElementType*,e.first);
      return elemastrk->trackType(ConvType);
    }
  };

  template<PFBlockElement::Type keytype, 
	   PFBlockElement::Type valtype,
	   bool useConv=false>
  struct NotCloserToOther : public PFFlaggedElementMatcher {
    const reco::PFBlockElement* comp;
    const reco::PFBlockRef& block;
    const reco::PFBlock::LinkData& links;   
    const float EoPin_cut;
    NotCloserToOther(const reco::PFBlockRef& b,
		     const reco::PFBlock::LinkData& l,
		     const PFFlaggedElement* e,
		     const float EoPcut=1.0e6): comp(e->first), 
						block(b), 
						links(l),
						EoPin_cut(EoPcut) { 
    }
    NotCloserToOther(const reco::PFBlockRef& b,
		     const reco::PFBlock::LinkData& l,
		     const reco::PFBlockElement* e,
		     const float EoPcut=1.0e6): comp(e), 
						block(b), 
						links(l),
						EoPin_cut(EoPcut) {
    }
    bool operator () (const PFFlaggedElement& e) {        
      if( !e.second || valtype != e.first->type() ) return false;      
      return elementNotCloserToOther<useConv>(block,
					      keytype,comp->index(),
					      valtype,e.first->index(),
					      EoPin_cut);
    }
  };

  struct LesserByDistance : public PFFlaggedElementSorter {
    const reco::PFBlockElement* comp;
    const reco::PFBlockRef& block;
    const reco::PFBlock::LinkData& links; 
    LesserByDistance(const reco::PFBlockRef& b,
		     const reco::PFBlock::LinkData& l,
		     const PFFlaggedElement* e): comp(e->first), 
						 block(b), 
						 links(l) {}
    LesserByDistance(const reco::PFBlockRef& b,
		     const reco::PFBlock::LinkData& l,
		     const reco::PFBlockElement* e): comp(e), 
						     block(b), 
						     links(l) {}
    bool operator () (const PFFlaggedElement& e1,
		      const PFFlaggedElement& e2) {                   
      double dist1 = block->dist(comp->index(), 
				 e1.first->index(),
				 links,
				 reco::PFBlock::LINKTEST_ALL);   
      double dist2 = block->dist(comp->index(), 
				 e2.first->index(),
				 links,
				 reco::PFBlock::LINKTEST_ALL);   
      dist1 = ( dist1 == -1.0 ? 1e6 : dist1 );
      dist2 = ( dist2 == -1.0 ? 1e6 : dist2 );
      return dist1 < dist2;
    }
  };
  
  bool isROLinkedByClusterOrTrack(const PFEGammaAlgo::ProtoEGObject& RO1,
				  const PFEGammaAlgo::ProtoEGObject& RO2 ) {
    // don't allow EB/EE to mix (11 Sept 2013)
    if( RO1.ecalclusters.size() && RO2.ecalclusters.size() ) {
      if(RO1.ecalclusters.front().first->clusterRef()->layer() !=
	 RO2.ecalclusters.front().first->clusterRef()->layer() ) {
	return false;
      }
    }
    const reco::PFBlockRef& blk = RO1.parentBlock;    
    bool not_closer;
    // check links track -> cluster
    for( const auto& cluster: RO1.ecalclusters ) {
      for( const auto& primgsf : RO2.primaryGSFs ) {
	not_closer = 
	  elementNotCloserToOther(blk,
				  cluster.first->type(),
				  cluster.first->index(),
				  primgsf.first->type(),
				  primgsf.first->index());
	if( not_closer ) return true;	  
      }
      for( const auto& primkf : RO2.primaryKFs) {
	not_closer = 
	  elementNotCloserToOther(blk,
				  cluster.first->type(),
				  cluster.first->index(),
				  primkf.first->type(),
				  primkf.first->index());
	if( not_closer ) return true;
      }
      for( const auto& secdkf : RO2.secondaryKFs) {
	not_closer = 
	    elementNotCloserToOther(blk,
				    cluster.first->type(),
				    cluster.first->index(),
				    secdkf.first->type(),
				    secdkf.first->index());
	if( not_closer ) return true;	  
      }
    }    
    // check links primary gsf -> secondary kf
    for( const auto& primgsf : RO1.primaryGSFs ) {      
      for( const auto& secdkf : RO2.secondaryKFs) {
	not_closer = 
	    elementNotCloserToOther(blk,
				    primgsf.first->type(),
				    primgsf.first->index(),
				    secdkf.first->type(),
				    secdkf.first->index());
	if( not_closer ) return true;	  
      }
    }
    // check links primary kf -> secondary kf
    for( const auto& primkf : RO1.primaryKFs ) {      
      for( const auto& secdkf : RO2.secondaryKFs) {
	not_closer = 
	    elementNotCloserToOther(blk,
				    primkf.first->type(),
				    primkf.first->index(),
				    secdkf.first->type(),
				    secdkf.first->index());
	if( not_closer ) return true;	  
      }
    }
    // check links secondary kf -> secondary kf
    for( const auto& secdkf1 : RO1.secondaryKFs ) {           
      for( const auto& secdkf2 : RO2.secondaryKFs) {
	not_closer = 
	    elementNotCloserToOther<true>(blk,
					  secdkf1.first->type(),
					  secdkf1.first->index(),
					  secdkf2.first->type(),
					  secdkf2.first->index());
	if( not_closer ) return true;	  
      }
    }
    return false;
  }
  
  struct TestIfROMergableByLink : public POMatcher {
    const PFEGammaAlgo::ProtoEGObject& comp;
    TestIfROMergableByLink(const PFEGammaAlgo::ProtoEGObject& RO) :
      comp(RO) {}
    bool operator() (const PFEGammaAlgo::ProtoEGObject& ro) {      
      return ( isROLinkedByClusterOrTrack(comp,ro) || 
	       isROLinkedByClusterOrTrack(ro,comp)   );      
    }
  }; 
}

PFEGammaAlgo::
PFEGammaAlgo(const PFEGammaAlgo::PFEGConfigInfo& cfg) : 
  cfg_(cfg),
  isvalid_(false), 
  verbosityLevel_(Silent), 
  nlost(0.0), nlayers(0.0),
  chi2(0.0), STIP(0.0), del_phi(0.0),HoverPt(0.0), EoverPt(0.0), track_pt(0.0),
  mvaValue(0.0),
  CrysPhi_(0.0), CrysEta_(0.0),  VtxZ_(0.0), ClusPhi_(0.0), ClusEta_(0.0),
  ClusR9_(0.0), Clus5x5ratio_(0.0),  PFCrysEtaCrack_(0.0), logPFClusE_(0.0), e3x3_(0.0),
  CrysIPhi_(0), CrysIEta_(0),
  CrysX_(0.0), CrysY_(0.0),
  EB(0.0),
  eSeed_(0.0), e1x3_(0.0),e3x1_(0.0), e1x5_(0.0), e2x5Top_(0.0),  e2x5Bottom_(0.0), e2x5Left_(0.0),  e2x5Right_(0.0),
  etop_(0.0), ebottom_(0.0), eleft_(0.0), eright_(0.0),
  e2x5Max_(0.0),
  PFPhoEta_(0.0), PFPhoPhi_(0.0), PFPhoR9_(0.0), PFPhoR9Corr_(0.0), SCPhiWidth_(0.0), SCEtaWidth_(0.0), 
  PFPhoEt_(0.0), RConv_(0.0), PFPhoEtCorr_(0.0), PFPhoE_(0.0), PFPhoECorr_(0.0), MustE_(0.0), E3x3_(0.0),
  dEta_(0.0), dPhi_(0.0), LowClusE_(0.0), RMSAll_(0.0), RMSMust_(0.0), nPFClus_(0.0),
  TotPS1_(0.0), TotPS2_(0.0),
  nVtx_(0.0),
  x0inner_(0.0), x0middle_(0.0), x0outer_(0.0),
  excluded_(0.0), Mustache_EtRatio_(0.0), Mustache_Et_out_(0.0)
{  
  
  // Set the tmva reader for electrons
  tmvaReaderEle_ = new TMVA::Reader("!Color:Silent");
  tmvaReaderEle_->AddVariable("lnPt_gsf",&lnPt_gsf);
  tmvaReaderEle_->AddVariable("Eta_gsf",&Eta_gsf);
  tmvaReaderEle_->AddVariable("dPtOverPt_gsf",&dPtOverPt_gsf);
  tmvaReaderEle_->AddVariable("DPtOverPt_gsf",&DPtOverPt_gsf);
  //tmvaReaderEle_->AddVariable("nhit_gsf",&nhit_gsf);
  tmvaReaderEle_->AddVariable("chi2_gsf",&chi2_gsf);
  //tmvaReaderEle_->AddVariable("DPtOverPt_kf",&DPtOverPt_kf);
  tmvaReaderEle_->AddVariable("nhit_kf",&nhit_kf);
  tmvaReaderEle_->AddVariable("chi2_kf",&chi2_kf);
  tmvaReaderEle_->AddVariable("EtotPinMode",&EtotPinMode);
  tmvaReaderEle_->AddVariable("EGsfPoutMode",&EGsfPoutMode);
  tmvaReaderEle_->AddVariable("EtotBremPinPoutMode",&EtotBremPinPoutMode);
  tmvaReaderEle_->AddVariable("DEtaGsfEcalClust",&DEtaGsfEcalClust);
  tmvaReaderEle_->AddVariable("SigmaEtaEta",&SigmaEtaEta);
  tmvaReaderEle_->AddVariable("HOverHE",&HOverHE);
//   tmvaReaderEle_->AddVariable("HOverPin",&HOverPin);
  tmvaReaderEle_->AddVariable("lateBrem",&lateBrem);
  tmvaReaderEle_->AddVariable("firstBrem",&firstBrem);
  tmvaReaderEle_->BookMVA("BDT",cfg_.mvaWeightFileEleID.c_str());
  
  
  //Book MVA  
  tmvaReader_ = new TMVA::Reader("!Color:Silent");  
  tmvaReader_->AddVariable("del_phi",&del_phi);  
  tmvaReader_->AddVariable("nlayers", &nlayers);  
  tmvaReader_->AddVariable("chi2",&chi2);  
  tmvaReader_->AddVariable("EoverPt",&EoverPt);  
  tmvaReader_->AddVariable("HoverPt",&HoverPt);  
  tmvaReader_->AddVariable("track_pt", &track_pt);  
  tmvaReader_->AddVariable("STIP",&STIP);  
  tmvaReader_->AddVariable("nlost", &nlost);  
  tmvaReader_->BookMVA("BDT",cfg_.mvaweightfile.c_str());  

  //Material Map
  TFile *XO_File = new TFile(cfg_.X0_Map.c_str(),"READ");
  X0_sum    = (TH2D*)XO_File->Get("TrackerSum");
  X0_inner  = (TH2D*)XO_File->Get("Inner");
  X0_middle = (TH2D*)XO_File->Get("Middle");
  X0_outer  = (TH2D*)XO_File->Get("Outer");
  
}

void PFEGammaAlgo::RunPFEG(const reco::PFBlockRef&  blockRef,
			      std::vector<bool>& active) {  

  fifthStepKfTrack_.clear();
  convGsfTrack_.clear();
  
  egCandidate_.clear();
  egExtra_.clear();
 
  // define how much is printed out for debugging.
  // ... will be setable via CFG file parameter
  verbosityLevel_ = Chatty;          // Chatty mode.
  
  buildAndRefineEGObjects(blockRef);
}

float PFEGammaAlgo::EvaluateResMVA(const reco::PFCandidate& photon, 
		    const std::vector<reco::CaloCluster>& PFClusters) {
  float BDTG=1;
  PFPhoEta_=photon.eta();
  PFPhoPhi_=photon.phi();
  PFPhoE_=photon.energy();
  //fill Material Map:
  int ix = X0_sum->GetXaxis()->FindBin(PFPhoEta_);
  int iy = X0_sum->GetYaxis()->FindBin(PFPhoPhi_);
  x0inner_= X0_inner->GetBinContent(ix,iy);
  x0middle_=X0_middle->GetBinContent(ix,iy);
  x0outer_=X0_outer->GetBinContent(ix,iy);
  SCPhiWidth_=photon.superClusterRef()->phiWidth();
  SCEtaWidth_=photon.superClusterRef()->etaWidth();
  Mustache Must;
  std::vector<unsigned int>insideMust;
  std::vector<unsigned int>outsideMust;
  std::multimap<float, unsigned int>OrderedClust;
  Must.FillMustacheVar(PFClusters);
  MustE_=Must.MustacheE();
  LowClusE_=Must.LowestMustClust();
  PFPhoR9Corr_=E3x3_/MustE_;
  Must.MustacheClust(PFClusters,insideMust, outsideMust );
  for(unsigned int i=0; i<insideMust.size(); ++i){
    int index=insideMust[i];
    OrderedClust.insert(make_pair(PFClusters[index].energy(),index));
  }
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second;
  std::multimap<float, unsigned int>::reverse_iterator rit;
  rit=OrderedClust.rbegin();
  unsigned int highEindex=(*rit).second;
  if(insideMust.size()>1){
    dEta_=fabs(PFClusters[highEindex].eta()-PFClusters[lowEindex].eta());
    dPhi_=asin(PFClusters[highEindex].phi()-PFClusters[lowEindex].phi());
  }
  else{
    dEta_=0;
    dPhi_=0;
    LowClusE_=0;
  }
  //calculate RMS for All clusters and up until the Next to Lowest inside the Mustache
  RMSAll_=ClustersPhiRMS(PFClusters, PFPhoPhi_);
  std::vector<reco::CaloCluster>PFMustClusters;
  if(insideMust.size()>2){
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      if(index==lowEindex)continue;
      PFMustClusters.push_back(PFClusters[index]);
    }
  }
  else{
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      PFMustClusters.push_back(PFClusters[index]);
    }    
  }
  RMSMust_=ClustersPhiRMS(PFMustClusters, PFPhoPhi_);
  //then use cluster Width for just one PFCluster
  RConv_=310;
  PFCandidate::ElementsInBlocks eleInBlocks = photon.elementsInBlocks();
  for(unsigned i=0; i<eleInBlocks.size(); i++)
    {
      PFBlockRef blockRef = eleInBlocks[i].first;
      unsigned indexInBlock = eleInBlocks[i].second;
      const edm::OwnVector< reco::PFBlockElement >&  elements=eleInBlocks[i].first->elements();
      const reco::PFBlockElement& element = elements[indexInBlock];
      if(element.type()==reco::PFBlockElement::TRACK){
	float R=sqrt(element.trackRef()->innerPosition().X()*element.trackRef()->innerPosition().X()+element.trackRef()->innerPosition().Y()*element.trackRef()->innerPosition().Y());
	if(RConv_>R)RConv_=R;
      }
      else continue;
    }
  float GC_Var[17];
  GC_Var[0]=PFPhoEta_;
  GC_Var[1]=PFPhoEt_;
  GC_Var[2]=PFPhoR9Corr_;
  GC_Var[3]=PFPhoPhi_;
  GC_Var[4]=SCEtaWidth_;
  GC_Var[5]=SCPhiWidth_;
  GC_Var[6]=x0inner_;  
  GC_Var[7]=x0middle_;
  GC_Var[8]=x0outer_;
  GC_Var[9]=RConv_;
  GC_Var[10]=LowClusE_;
  GC_Var[11]=RMSMust_;
  GC_Var[12]=RMSAll_;
  GC_Var[13]=dEta_;
  GC_Var[14]=dPhi_;
  GC_Var[15]=nVtx_;
  GC_Var[16]=MustE_;
  
  BDTG=ReaderRes_->GetResponse(GC_Var);
  //  cout<<"Res "<<BDTG<<endl;
  
  //  cout<<"BDTG Parameters X0"<<x0inner_<<", "<<x0middle_<<", "<<x0outer_<<endl;
  //  cout<<"Et, Eta, Phi "<<PFPhoEt_<<", "<<PFPhoEta_<<", "<<PFPhoPhi_<<endl;
  // cout<<"PFPhoR9 "<<PFPhoR9_<<endl;
  // cout<<"R "<<RConv_<<endl;
  
  return BDTG;
   
}

float PFEGammaAlgo::EvaluateGCorrMVA(const reco::PFCandidate& photon, 
			    const std::vector<CaloCluster>& PFClusters) {
  float BDTG=1;
  PFPhoEta_=photon.eta();
  PFPhoPhi_=photon.phi();
  PFPhoE_=photon.energy();
    //fill Material Map:
  int ix = X0_sum->GetXaxis()->FindBin(PFPhoEta_);
  int iy = X0_sum->GetYaxis()->FindBin(PFPhoPhi_);
  x0inner_= X0_inner->GetBinContent(ix,iy);
  x0middle_=X0_middle->GetBinContent(ix,iy);
  x0outer_=X0_outer->GetBinContent(ix,iy);
  SCPhiWidth_=photon.superClusterRef()->phiWidth();
  SCEtaWidth_=photon.superClusterRef()->etaWidth();
  Mustache Must;
  std::vector<unsigned int>insideMust;
  std::vector<unsigned int>outsideMust;
  std::multimap<float, unsigned int>OrderedClust;
  Must.FillMustacheVar(PFClusters);
  MustE_=Must.MustacheE();
  LowClusE_=Must.LowestMustClust();
  PFPhoR9Corr_=E3x3_/MustE_;
  Must.MustacheClust(PFClusters,insideMust, outsideMust );
  for(unsigned int i=0; i<insideMust.size(); ++i){
    int index=insideMust[i];
    OrderedClust.insert(make_pair(PFClusters[index].energy(),index));
  }
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second;
  std::multimap<float, unsigned int>::reverse_iterator rit;
  rit=OrderedClust.rbegin();
  unsigned int highEindex=(*rit).second;
  if(insideMust.size()>1){
    dEta_=fabs(PFClusters[highEindex].eta()-PFClusters[lowEindex].eta());
    dPhi_=asin(PFClusters[highEindex].phi()-PFClusters[lowEindex].phi());
  }
  else{
    dEta_=0;
    dPhi_=0;
    LowClusE_=0;
  }
  //calculate RMS for All clusters and up until the Next to Lowest inside the Mustache
  RMSAll_=ClustersPhiRMS(PFClusters, PFPhoPhi_);
  std::vector<reco::CaloCluster>PFMustClusters;
  if(insideMust.size()>2){
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      if(index==lowEindex)continue;
      PFMustClusters.push_back(PFClusters[index]);
    }
  }
  else{
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      PFMustClusters.push_back(PFClusters[index]);
    }    
  }
  RMSMust_=ClustersPhiRMS(PFMustClusters, PFPhoPhi_);
  //then use cluster Width for just one PFCluster
  RConv_=310;
  PFCandidate::ElementsInBlocks eleInBlocks = photon.elementsInBlocks();
  for(unsigned i=0; i<eleInBlocks.size(); i++)
    {
      PFBlockRef blockRef = eleInBlocks[i].first;
      unsigned indexInBlock = eleInBlocks[i].second;
      const edm::OwnVector< reco::PFBlockElement >&  elements=eleInBlocks[i].first->elements();
      const reco::PFBlockElement& element = elements[indexInBlock];
      if(element.type()==reco::PFBlockElement::TRACK){
	float R=sqrt(element.trackRef()->innerPosition().X()*element.trackRef()->innerPosition().X()+element.trackRef()->innerPosition().Y()*element.trackRef()->innerPosition().Y());
	if(RConv_>R)RConv_=R;
      }
      else continue;
    }
  //cout<<"Nvtx "<<nVtx_<<endl;
  if(fabs(PFPhoEta_)<1.4446){
    float GC_Var[17];
    GC_Var[0]=PFPhoEta_;
    GC_Var[1]=PFPhoECorr_;
    GC_Var[2]=PFPhoR9Corr_;
    GC_Var[3]=SCEtaWidth_;
    GC_Var[4]=SCPhiWidth_;
    GC_Var[5]=PFPhoPhi_;
    GC_Var[6]=x0inner_;
    GC_Var[7]=x0middle_;
    GC_Var[8]=x0outer_;
    GC_Var[9]=RConv_;
    GC_Var[10]=LowClusE_;
    GC_Var[11]=RMSMust_;
    GC_Var[12]=RMSAll_;
    GC_Var[13]=dEta_;
    GC_Var[14]=dPhi_;
    GC_Var[15]=nVtx_;
    GC_Var[16]=MustE_;
    BDTG=ReaderGCEB_->GetResponse(GC_Var);
  }
  else if(PFPhoR9_>0.94){
    float GC_Var[19];
    GC_Var[0]=PFPhoEta_;
    GC_Var[1]=PFPhoECorr_;
    GC_Var[2]=PFPhoR9Corr_;
    GC_Var[3]=SCEtaWidth_;
    GC_Var[4]=SCPhiWidth_;
    GC_Var[5]=PFPhoPhi_;
    GC_Var[6]=x0inner_;
    GC_Var[7]=x0middle_;
    GC_Var[8]=x0outer_;
    GC_Var[9]=RConv_;
    GC_Var[10]=LowClusE_;
    GC_Var[11]=RMSMust_;
    GC_Var[12]=RMSAll_;
    GC_Var[13]=dEta_;
    GC_Var[14]=dPhi_;
    GC_Var[15]=nVtx_;
    GC_Var[16]=TotPS1_;
    GC_Var[17]=TotPS2_;
    GC_Var[18]=MustE_;
    BDTG=ReaderGCEEhR9_->GetResponse(GC_Var);
  }
  
  else{
    float GC_Var[19];
    GC_Var[0]=PFPhoEta_;
    GC_Var[1]=PFPhoE_;
    GC_Var[2]=PFPhoR9Corr_;
    GC_Var[3]=SCEtaWidth_;
    GC_Var[4]=SCPhiWidth_;
    GC_Var[5]=PFPhoPhi_;
    GC_Var[6]=x0inner_;
    GC_Var[7]=x0middle_;
    GC_Var[8]=x0outer_;
    GC_Var[9]=RConv_;
    GC_Var[10]=LowClusE_;
    GC_Var[11]=RMSMust_;
    GC_Var[12]=RMSAll_;
    GC_Var[13]=dEta_;
    GC_Var[14]=dPhi_;
    GC_Var[15]=nVtx_;
    GC_Var[16]=TotPS1_;
    GC_Var[17]=TotPS2_;
    GC_Var[18]=MustE_;
    BDTG=ReaderGCEElR9_->GetResponse(GC_Var);
  }
  //cout<<"GC "<<BDTG<<endl;

  return BDTG;
  
}

double PFEGammaAlgo::
ClustersPhiRMS(const std::vector<reco::CaloCluster>& PFClusters, 
	       float PFPhoPhi) const {
  double PFClustPhiRMS=0;
  double delPhi2=0;
  double delPhiSum=0;
  double ClusSum=0;
  for(unsigned int c=0; c<PFClusters.size(); ++c){
    delPhi2=(acos(cos(PFPhoPhi-PFClusters[c].phi()))* acos(cos(PFPhoPhi-PFClusters[c].phi())) )+delPhi2;
    delPhiSum=delPhiSum+ acos(cos(PFPhoPhi-PFClusters[c].phi()))*PFClusters[c].energy();
    ClusSum=ClusSum+PFClusters[c].energy();
  }
  double meandPhi=delPhiSum/ClusSum;
  PFClustPhiRMS=sqrt(fabs(delPhi2/ClusSum - (meandPhi*meandPhi)));
  
  return PFClustPhiRMS;
}

float PFEGammaAlgo::
EvaluateLCorrMVA(const reco::PFClusterRef& clusterRef ) {
  float BDTG=1;
  PFPhotonClusters ClusterVar(clusterRef);
  std::pair<double, double>ClusCoor=ClusterVar.GetCrysCoor();
  std::pair<int, int>ClusIndex=ClusterVar.GetCrysIndex();
  //Local Coordinates:
  if(clusterRef->layer()==PFLayer:: ECAL_BARREL ){//is Barrel
    PFCrysEtaCrack_=ClusterVar.EtaCrack();
    CrysEta_=ClusCoor.first;
    CrysPhi_=ClusCoor.second;
    CrysIEta_=ClusIndex.first;
    CrysIPhi_=ClusIndex.second;
  }
  else{
    CrysX_=ClusCoor.first;
    CrysY_=ClusCoor.second;
  }
  //Shower Shape Variables:
  eSeed_= ClusterVar.E5x5Element(0, 0)/clusterRef->energy();
  etop_=ClusterVar.E5x5Element(0,1)/clusterRef->energy();
  ebottom_=ClusterVar.E5x5Element(0,-1)/clusterRef->energy();
  eleft_=ClusterVar.E5x5Element(-1,0)/clusterRef->energy();
  eright_=ClusterVar.E5x5Element(1,0)/clusterRef->energy();
  e1x3_=(ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(0,-1))/clusterRef->energy();
  e3x1_=(ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(-1,0)+ClusterVar.E5x5Element(1,0))/clusterRef->energy();
  e1x5_=ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(0,-2)+ClusterVar.E5x5Element(0,-1)+ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(0,2);
  
  e2x5Top_=(ClusterVar.E5x5Element(-2,2)+ClusterVar.E5x5Element(-1, 2)+ClusterVar.E5x5Element(0, 2)
	    +ClusterVar.E5x5Element(1, 2)+ClusterVar.E5x5Element(2, 2)
	    +ClusterVar.E5x5Element(-2,1)+ClusterVar.E5x5Element(-1,1)+ClusterVar.E5x5Element(0,1)
	    +ClusterVar.E5x5Element(1,1)+ClusterVar.E5x5Element(2,1))/clusterRef->energy();
  e2x5Bottom_=(ClusterVar.E5x5Element(-2,-2)+ClusterVar.E5x5Element(-1,-2)+ClusterVar.E5x5Element(0,-2)
	       +ClusterVar.E5x5Element(1,-2)+ClusterVar.E5x5Element(2,-2)
	       +ClusterVar.E5x5Element(-2,1)+ClusterVar.E5x5Element(-1,1)
	       +ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(1,1)+ClusterVar.E5x5Element(2,1))/clusterRef->energy();
  e2x5Left_= (ClusterVar.E5x5Element(-2,-2)+ClusterVar.E5x5Element(-2,-1)
	      +ClusterVar.E5x5Element(-2,0)
	       +ClusterVar.E5x5Element(-2,1)+ClusterVar.E5x5Element(-2,2)
	      +ClusterVar.E5x5Element(-1,-2)+ClusterVar.E5x5Element(-1,-1)+ClusterVar.E5x5Element(-1,0)
	      +ClusterVar.E5x5Element(-1,1)+ClusterVar.E5x5Element(-1,2))/clusterRef->energy();
  
  e2x5Right_ =(ClusterVar.E5x5Element(2,-2)+ClusterVar.E5x5Element(2,-1)
	       +ClusterVar.E5x5Element(2,0)+ClusterVar.E5x5Element(2,1)+ClusterVar.E5x5Element(2,2)
	       +ClusterVar.E5x5Element(1,-2)+ClusterVar.E5x5Element(1,-1)+ClusterVar.E5x5Element(1,0)
	       +ClusterVar.E5x5Element(1,1)+ClusterVar.E5x5Element(1,2))/clusterRef->energy();
  float centerstrip=ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(0, -2)
    +ClusterVar.E5x5Element(0,-1)+ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(0,2);
  float rightstrip=ClusterVar.E5x5Element(1, 0)+ClusterVar.E5x5Element(1,1)
    +ClusterVar.E5x5Element(1,2)+ClusterVar.E5x5Element(1,-1)+ClusterVar.E5x5Element(1,-2);
  float leftstrip=ClusterVar.E5x5Element(-1,0)+ClusterVar.E5x5Element(-1,-1)+ClusterVar.E5x5Element(-1,2)
    +ClusterVar.E5x5Element(-1,1)+ClusterVar.E5x5Element(-1,2);
  
  if(rightstrip>leftstrip)e2x5Max_=rightstrip+centerstrip;
  else e2x5Max_=leftstrip+centerstrip;
  e2x5Max_=e2x5Max_/clusterRef->energy();
  //GetCrysCoordinates(clusterRef);
  //fill5x5Map(clusterRef);
  VtxZ_=cfg_.primaryVtx->z();
  ClusPhi_=clusterRef->position().phi(); 
  ClusEta_=fabs(clusterRef->position().eta());
  EB=fabs(clusterRef->position().eta())/clusterRef->position().eta();
  logPFClusE_=log(clusterRef->energy());
  if(ClusEta_<1.4446){
    float LC_Var[26];
    LC_Var[0]=VtxZ_;
    LC_Var[1]=EB;
    LC_Var[2]=ClusEta_;
    LC_Var[3]=ClusPhi_;
    LC_Var[4]=logPFClusE_;
    LC_Var[5]=eSeed_;
    //top bottom left right
    LC_Var[6]=etop_;
    LC_Var[7]=ebottom_;
    LC_Var[8]=eleft_;
    LC_Var[9]=eright_;
    LC_Var[10]=ClusR9_;
    LC_Var[11]=e1x3_;
    LC_Var[12]=e3x1_;
    LC_Var[13]=Clus5x5ratio_;
    LC_Var[14]=e1x5_;
    LC_Var[15]=e2x5Max_;
    LC_Var[16]=e2x5Top_;
    LC_Var[17]=e2x5Bottom_;
    LC_Var[18]=e2x5Left_;
    LC_Var[19]=e2x5Right_;
    LC_Var[20]=CrysEta_;
    LC_Var[21]=CrysPhi_;
    float CrysIphiMod2=CrysIPhi_%2;
    float CrysIetaMod5=CrysIEta_%5;
    float CrysIphiMod20=CrysIPhi_%20;
    LC_Var[22]=CrysIphiMod2;
    LC_Var[23]=CrysIetaMod5;
    LC_Var[24]=CrysIphiMod20;   
    LC_Var[25]=PFCrysEtaCrack_;
    BDTG=ReaderLCEB_->GetResponse(LC_Var);   
    //cout<<"LC "<<BDTG<<endl;  
  }
  else{
    float LC_Var[22];
    LC_Var[0]=VtxZ_;
    LC_Var[1]=EB;
    LC_Var[2]=ClusEta_;
    LC_Var[3]=ClusPhi_;
    LC_Var[4]=logPFClusE_;
    LC_Var[5]=eSeed_;
    //top bottom left right
    LC_Var[6]=etop_;
    LC_Var[7]=ebottom_;
    LC_Var[8]=eleft_;
    LC_Var[9]=eright_;
    LC_Var[10]=ClusR9_;
    LC_Var[11]=e1x3_;
    LC_Var[12]=e3x1_;
    LC_Var[13]=Clus5x5ratio_;
    LC_Var[14]=e1x5_;
    LC_Var[15]=e2x5Max_;
    LC_Var[16]=e2x5Top_;
    LC_Var[17]=e2x5Bottom_;
    LC_Var[18]=e2x5Left_;
    LC_Var[19]=e2x5Right_;
    LC_Var[20]=CrysX_;
    LC_Var[21]=CrysY_;
    BDTG=ReaderLCEE_->GetResponse(LC_Var);   
    //cout<<"LC "<<BDTG<<endl;  
  }
   return BDTG;
  
}

bool PFEGammaAlgo::EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, 
					const reco::Vertex& primaryvtx, 
					unsigned int track_index) {  
  bool convtkfound=false;  
  const reco::PFBlock& block = *blockref;  
  const edm::OwnVector< reco::PFBlockElement >& elements = block.elements();  
  //use this to store linkdata in the associatedElements function below  
  PFBlock::LinkData linkData =  block.linkData();  
  //calculate MVA Variables  
  chi2=elements[track_index].trackRef()->chi2()/elements[track_index].trackRef()->ndof();  
  nlost=elements[track_index].trackRef()->trackerExpectedHitsInner().numberOfLostHits();  
  nlayers=elements[track_index].trackRef()->hitPattern().trackerLayersWithMeasurement();  
  track_pt=elements[track_index].trackRef()->pt();  
  STIP=elements[track_index].trackRefPF()->STIP();  
   
  float linked_e=0;  
  float linked_h=0;  
  std::multimap<double, unsigned int> ecalAssoTrack;  
  block.associatedElements( track_index,linkData,  
			    ecalAssoTrack,  
			    reco::PFBlockElement::ECAL,  
			    reco::PFBlock::LINKTEST_ALL );  
  std::multimap<double, unsigned int> hcalAssoTrack;  
  block.associatedElements( track_index,linkData,  
			    hcalAssoTrack,  
			    reco::PFBlockElement::HCAL,  
			    reco::PFBlock::LINKTEST_ALL );  
  if(ecalAssoTrack.size() > 0) {  
    for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoTrack.begin();  
	itecal != ecalAssoTrack.end(); ++itecal) {  
      linked_e=linked_e+elements[itecal->second].clusterRef()->energy();  
    }  
  }  
  if(hcalAssoTrack.size() > 0) {  
    for(std::multimap<double, unsigned int>::iterator ithcal = hcalAssoTrack.begin();  
	ithcal != hcalAssoTrack.end(); ++ithcal) {  
      linked_h=linked_h+elements[ithcal->second].clusterRef()->energy();  
    }  
  }  
  EoverPt=linked_e/elements[track_index].trackRef()->pt();  
  HoverPt=linked_h/elements[track_index].trackRef()->pt();  
  GlobalVector rvtx(elements[track_index].trackRef()->innerPosition().X()-primaryvtx.x(),  
		    elements[track_index].trackRef()->innerPosition().Y()-primaryvtx.y(),  
		    elements[track_index].trackRef()->innerPosition().Z()-primaryvtx.z());  
  double vtx_phi=rvtx.phi();  
  //delta Phi between conversion vertex and track  
  del_phi=fabs(deltaPhi(vtx_phi, elements[track_index].trackRef()->innerMomentum().Phi()));  
  mvaValue = tmvaReader_->EvaluateMVA("BDT");  
  if(mvaValue > cfg_.mvaConvCut) convtkfound=true;  
  return convtkfound;  
}

//Recover Early Conversions reconstructed as PFelectrons
void PFEGammaAlgo::EarlyConversion(    
				   //std::auto_ptr< reco::PFCandidateCollection > 
				   //&pfElectronCandidates_,
				   const std::vector<reco::PFCandidate>& 
				   tempElectronCandidates,
				   const reco::PFBlockElementSuperCluster* sc
				   ) {
  //step 1 check temp electrons for clusters that match Photon Supercluster:
  // permElectronCandidates->clear();
  int count=0;
  for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec ) 
    {
      //      bool matched=false;
      int mh=ec->gsfTrackRef()->trackerExpectedHitsInner().numberOfLostHits();
      //if(mh==0)continue;//Case where missing hits greater than zero
      
      reco::GsfTrackRef gsf=ec->gsfTrackRef();
      //some hoopla to get Electron SC ref
      
      if(gsf->extra().isAvailable() && gsf->extra()->seedRef().isAvailable() && mh>0) 
	{
	  reco::ElectronSeedRef seedRef=  gsf->extra()->seedRef().castTo<reco::ElectronSeedRef>();
	  if(seedRef.isAvailable() && seedRef->isEcalDriven()) 
	    {
	      reco::SuperClusterRef ElecscRef = seedRef->caloCluster().castTo<reco::SuperClusterRef>();
	      
	      if(ElecscRef.isNonnull()){
		//finally see if it matches:
		reco::SuperClusterRef PhotscRef=sc->superClusterRef();
		if(PhotscRef==ElecscRef)
		  {
		    match_ind.push_back(count);
		    //  matched=true; 
		    //cout<<"Matched Electron with Index "<<count<<" This is the electron "<<*ec<<endl;
		    //find that they have the same SC footprint start to collect Clusters and tracks and these will be passed to PFPhoton
		    reco::PFCandidate::ElementsInBlocks eleInBlocks = ec->elementsInBlocks();
		    for(unsigned i=0; i<eleInBlocks.size(); i++) 
		      {
			reco::PFBlockRef blockRef = eleInBlocks[i].first;
			unsigned indexInBlock = eleInBlocks[i].second;	 
			//const edm::OwnVector< reco::PFBlockElement >&  elements=eleInBlocks[i].first->elements();
			//const reco::PFBlockElement& element = elements[indexInBlock];  		
			
			AddFromElectron_.push_back(indexInBlock);	       	
		      }		    
		  }		
	      }
	    }	  
	}           
      count++;
    }
}

bool PFEGammaAlgo::isAMuon(const reco::PFBlockElement& pfbe) {
  NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::TRACK>
    getTrackPartner(_currentblock,_currentlinks,&pfbe);
  switch( pfbe.type() ) {
  case reco::PFBlockElement::GSF:    
    {
      auto& tracks = _splayedblock[reco::PFBlockElement::TRACK];
      auto notmatched = 
	std::partition(tracks.begin(),tracks.end(),getTrackPartner);
      for( auto tk = tracks.begin(); tk != notmatched; ++tk ) {
	if( PFMuonAlgo::isMuon(*(tk->first)) ) return true;
      }
    }
    break;
  case reco::PFBlockElement::TRACK:
    return PFMuonAlgo::isMuon(pfbe);
    break;
  default:
    break;
  }
  return false;
}

void PFEGammaAlgo::buildAndRefineEGObjects(const reco::PFBlockRef& block) {
  LOGVERB("PFEGammaAlgo") 
    << "Resetting PFEGammaAlgo for new block and running!" << std::endl;
  _splayedblock.clear();
  _recoveredlinks.clear();
  _refinableObjects.clear();
  _finalCandidates.clear();  
  _splayedblock.resize(12); // make sure that we always have the SC entry

  _currentblock = block;
  _currentlinks = block->linkData();
  LOGDRESSED("PFEGammaAlgo") << *_currentblock << std::endl;
  LOGVERB("PFEGammaAlgo") << "Splaying block" << std::endl;  
  //unwrap the PF block into a fast access map
  for( const auto& pfelement : _currentblock->elements() ) {
    if( isAMuon(pfelement) ) continue; // don't allow muons in our element list
    const size_t itype = (size_t)pfelement.type();    
    if( itype >= _splayedblock.size() ) _splayedblock.resize(itype+1);    
    _splayedblock[itype].push_back(std::make_pair(&pfelement,true));    
  }

  // show the result of splaying the tree if it's really *really* needed
#ifdef PFLOW_DEBUG
  std::stringstream splayout;
  for( size_t itype = 0; itype < _splayedblock.size(); ++itype ) {
    splayout << "\tType: " << itype << " indices: ";
    for( const auto& flaggedelement : _splayedblock[itype] ) {
      splayout << flaggedelement.first->index() << ' ';
    }
    if( itype != _splayedblock.size() - 1 ) splayout << std::endl;
  }
  LOGVERB("PFEGammaAlgo") << splayout.str();  
#endif

  // precleaning of the ECAL clusters with respect to primary KF tracks
  // we don't allow clusters in super clusters to be locked out this way
  removeOrLinkECALClustersToKFTracks();

  initializeProtoCands(_refinableObjects); 
  LOGDRESSED("PFEGammaAlgo") 
    << "Initialized " << _refinableObjects.size() << " proto-EGamma objects"
    << std::endl;
  dumpCurrentRefinableObjects();

  //
  // now we start the refining steps
  //
  //
  
  // --- Primary Linking Step ---
  // since this is particle flow and we try to work from the pixels out
  // we start by linking the tracks together and finding the ECAL clusters
  for( auto& RO : _refinableObjects ) {
    // find the KF tracks associated to GSF primary tracks
    linkRefinableObjectGSFTracksToKFs(RO);
    // link secondary KF tracks associated to primary KF tracks
    linkRefinableObjectPrimaryKFsToSecondaryKFs(RO);
    // pick up clusters that are linked to the GSF primary
    linkRefinableObjectPrimaryGSFTrackToECAL(RO);
    // link associated KF to ECAL (ECAL part grabs PS clusters too if able)
    linkRefinableObjectKFTracksToECAL(RO);
    // now finally look for clusters associated to brem tangents
    linkRefinableObjectBremTangentsToECAL(RO);
  }

  LOGDRESSED("PFEGammaAlgo")
    << "Dumping after GSF and KF Track (Primary) Linking : " << std::endl;
  dumpCurrentRefinableObjects();

  // merge objects after primary linking
  mergeROsByAnyLink(_refinableObjects);

  LOGDRESSED("PFEGammaAlgo")
    << "Dumping after first merging operation : " << std::endl;
  dumpCurrentRefinableObjects();

  // --- Secondary Linking Step ---
  // after this we go through the ECAL clusters on the remaining tracks
  // and try to link those in...
  for( auto& RO : _refinableObjects ) {    
    // look for conversion legs
    linkRefinableObjectECALToSingleLegConv(RO);
    dumpCurrentRefinableObjects();
    // look for tracks that complement conversion legs
    linkRefinableObjectConvSecondaryKFsToSecondaryKFs(RO);
    // look again for ECAL clusters (this time with an e/p cut)
    linkRefinableObjectSecondaryKFsToECAL(RO);
  } 

  LOGDRESSED("PFEGammaAlgo")
    << "Dumping after ECAL to Track (Secondary) Linking : " << std::endl;
  dumpCurrentRefinableObjects();

  // merge objects after primary linking
  mergeROsByAnyLink(_refinableObjects);

  LOGDRESSED("PFEGammaAlgo")
    << "There are " << _refinableObjects.size() 
    << " after the 2nd merging step." << std::endl;
  dumpCurrentRefinableObjects();

  // -- unlinking and proto-object vetos 
  for( auto& RO : _refinableObjects ) {
    // remove secondary KFs (and possibly ECALs) matched to HCAL clusters
    unlinkRefinableObjectKFandECALMatchedToHCAL(RO, false, false);
    // remove secondary KFs and ECALs linked to them that have bad E/p_in 
    // and spoil the resolution
    unlinkRefinableObjectKFandECALWithBadEoverP(RO);
  }

  LOGDRESSED("PFEGammaAlgo")
    << "There are " << _refinableObjects.size() 
    << " after the unlinking and vetos step." << std::endl;
  dumpCurrentRefinableObjects();

  // fill the PF candidates and then build the refined SC
  fillPFCandidates(_refinableObjects,outcands_,outcandsextra_);

}

void PFEGammaAlgo::
initializeProtoCands(std::list<PFEGammaAlgo::ProtoEGObject>& egobjs) {
  // step 1: build SC based proto-candidates
  // in the future there will be an SC Et requirement made here to control
  // block size
  for( auto& element : _splayedblock[PFBlockElement::SC] ) {
    LOGDRESSED("PFEGammaAlgo") 
      << "creating SC-based proto-object" << std::endl
      << "\tSC at index: " << element.first->index() 
      << " has type: " << element.first->type() << std::endl;
    element.second = false;
    ProtoEGObject fromSC;
    fromSC.parentBlock = _currentblock;
    fromSC.parentSC = docast(const PFSCElement*,element.first);
    // splay the supercluster so we can knock out used elements
    bool sc_success = 
      unwrapSuperCluster(fromSC.parentSC,fromSC.ecalclusters,fromSC.ecal2ps);
    if( sc_success ) _refinableObjects.push_back(fromSC);
  }
  // step 2: build GSF-seed-based proto-candidates
  reco::GsfTrackRef gsfref_forextra;
  reco::TrackExtraRef gsftrk_extra;
  reco::ElectronSeedRef theseedref; 
  std::list<ProtoEGObject>::iterator objsbegin, objsend;  
  for( auto& element : _splayedblock[PFBlockElement::GSF] ) {
    LOGDRESSED("PFEGammaAlgo") 
      << "creating GSF-based proto-object" << std::endl
      << "\tGSF at index: " << element.first->index() 
      << " has type: " << element.first->type() << std::endl;
    const PFGSFElement* elementAsGSF = 
      docast(const PFGSFElement*,element.first);
    if( elementAsGSF->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) ) {
      continue; // for now, do not allow dedicated brems to make proto-objects
    }
    element.second = false;
    
    ProtoEGObject fromGSF;    
    gsfref_forextra = elementAsGSF->GsftrackRef();
    gsftrk_extra = ( gsfref_forextra.isAvailable() ? 
		     gsfref_forextra->extra() : reco::TrackExtraRef() );
    theseedref = ( gsftrk_extra.isAvailable() ?
		   gsftrk_extra->seedRef().castTo<reco::ElectronSeedRef>() :
		   reco::ElectronSeedRef() );  
    fromGSF.electronSeed = theseedref;
    // exception if there's no seed
    if(fromGSF.electronSeed.isNull() || !fromGSF.electronSeed.isAvailable()) {
      std::stringstream gsf_err;
      elementAsGSF->Dump(gsf_err,"\t");
      throw cms::Exception("PFEGammaAlgo::initializeProtoCands()")
	<< "Found a GSF track with no seed! This should not happen!" 
	<< std::endl << gsf_err.str() << std::endl;
    }
    // flag this GSF element as globally used and push back the track ref
    // into the protocand
    element.second = false;
    fromGSF.parentBlock = _currentblock;
    fromGSF.primaryGSFs.push_back(std::make_pair(elementAsGSF,true));
    // add the directly matched brem tangents    
    for( auto& brem : _splayedblock[PFBlockElement::BREM] ) {
      float dist = _currentblock->dist(elementAsGSF->index(),
				       brem.first->index(),
				       _currentlinks,
				       reco::PFBlock::LINKTEST_ALL);
      if( dist == 0.001f ) {
	const PFBremElement* eAsBrem = 
	  docast(const PFBremElement*,brem.first);
	fromGSF.brems.push_back(std::make_pair(eAsBrem,true));
	fromGSF.localMap.push_back( ElementMap::value_type(eAsBrem,elementAsGSF) );
	fromGSF.localMap.push_back( ElementMap::value_type(elementAsGSF,eAsBrem) );
	brem.second = false;
      }
    }
    // if this track is ECAL seeded reset links or import cluster
    // tracker (this is pixel only, right?) driven seeds just get the GSF
    // track associated since this only branches for ECAL Driven seeds
    if( fromGSF.electronSeed->isEcalDriven() ) {
      // step 2a: either merge with existing ProtoEG object with SC or add 
      //          SC directly to this proto EG object if not present
      LOGDRESSED("PFEGammaAlgo")
	<< "GSF-based proto-object is ECAL driven, merging SC-cand"
	<< std::endl;
      LOGVERB("PFEGammaAlgo")
	<< "ECAL Seed Ptr: " << fromGSF.electronSeed.get() 
	<< " isAvailable: " << fromGSF.electronSeed.isAvailable() 
	<< " isNonnull: " << fromGSF.electronSeed.isNonnull() 
	<< std::endl;           
      SeedMatchesToProtoObject sctoseedmatch(fromGSF.electronSeed);      
      objsbegin = _refinableObjects.begin();
      objsend   = _refinableObjects.end();
      // this auto is a std::list<ProtoEGObject>::iterator
      auto clusmatch = std::find_if(objsbegin,objsend,sctoseedmatch);
      if( clusmatch != objsend ) {
	fromGSF.parentSC = clusmatch->parentSC;
	fromGSF.ecalclusters = std::move(clusmatch->ecalclusters);
	fromGSF.ecal2ps  = std::move(clusmatch->ecal2ps);
	_refinableObjects.erase(clusmatch);	
      } else if (fromGSF.electronSeed.isAvailable()  && 
		 fromGSF.electronSeed.isNonnull()) {
	// link tests in the gap region can current split a gap electron
	// HEY THIS IS A WORK AROUND FOR A KNOWN BUG IN PFBLOCKALGO
	// MAYBE WE SHOULD FIX IT??????????????????????????????????
	LOGERR("PFEGammaAlgo")
	  << "Encountered the known GSF-SC splitting bug "
	  << " in PFBlockAlgo! We should really fix this!" << std::endl; 
      } else { // SC was not in a earlier proto-object	
	std::stringstream gsf_err;
	elementAsGSF->Dump(gsf_err,"\t");
	throw cms::Exception("PFEGammaAlgo::initializeProtoCands()")
	  << "Expected SuperCluster from ECAL driven GSF seed "
	  << "was not found in the block!" << std::endl 
	  << gsf_err.str() << std::endl;
      } // supercluster in block
    } // is ECAL driven seed?
    _refinableObjects.push_back(fromGSF);
  } // end loop on GSF elements of block
}

bool PFEGammaAlgo::
unwrapSuperCluster(const PFSCElement* thesc,
		   std::vector<PFClusterFlaggedElement>& ecalclusters,
		   ClusterMap& ecal2ps) {
  ecalclusters.clear();
  ecal2ps.clear();
  LOGVERB("PFEGammaAlgo")
    << "Pointer to SC element: 0x" 
    << std::hex << thesc << std::dec << std::endl
    << "cleared ecalclusters and ecal2ps!" << std::endl;  
  auto ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ecalend = _splayedblock[reco::PFBlockElement::ECAL].end();  
  if( ecalbegin == ecalend ) {
    LOGERR("PFEGammaAlgo::unwrapSuperCluster()")
      << "There are no ECAL elements in a block with imported SC!" 
      << " This is a bug we should fix this!" 
      << std::endl;
    return false;
  }
  reco::SuperClusterRef scref = thesc->superClusterRef();
  // this check needs to be done in a different way
  const bool is_pf_sc = (bool)
    dynamic_cast<const reco::PFCluster*>((*scref->clustersBegin()).get());
  if( !(scref.isAvailable() && scref.isNonnull()) ) {
    throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
      << "SuperCluster pointed to by block element is null!" 
      << std::endl;
  }
  LOGDRESSED("PFEGammaAlgo")
    << "Got a valid super cluster ref! 0x" 
    << std::hex << scref.get() << std::dec << std::endl;
  const size_t nscclusters = scref->clustersSize();
  const size_t nscpsclusters = scref->preshowerClustersSize();
  size_t npfpsclusters = 0;
  size_t npfclusters = 0;
  LOGDRESSED("PFEGammaAlgo")
    << "Precalculated cluster multiplicities: " 
    << nscclusters << ' ' << nscpsclusters << std::endl;
  NotCloserToOther<reco::PFBlockElement::SC,reco::PFBlockElement::ECAL> 
    ecalClustersInSC(_currentblock,_currentlinks,thesc);
  auto firstnotinsc = std::partition(ecalbegin,ecalend,ecalClustersInSC);
  //reset the begin and end iterators
  ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  ecalend = _splayedblock[reco::PFBlockElement::ECAL].end();  
  if( firstnotinsc == ecalbegin ) {
    LOGERR("PFEGammaAlgo::unwrapSuperCluster()")
      << "No associated block elements to SuperCluster!" 
      << " This is a bug we should fix!"
      << std::endl;
    return false;
  }
  npfclusters = std::distance(ecalbegin,firstnotinsc);
  // ensure we have found the correct number of PF ecal clusters in the case
  // that this is a PF supercluster, otherwise all bets are off
  if( is_pf_sc && nscclusters != npfclusters ) {
    std::stringstream sc_err;
    thesc->Dump(sc_err,"\t");
    throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
      << "The number of found ecal elements ("
      << nscclusters << ") in block is not the same as"
      << " the number of ecal PF clusters reported by the PFSuperCluster"
      << " itself (" << npfclusters
      << ")! This should not happen!" << std::endl 
      << sc_err.str() << std::endl;
  }
  for( auto ecalitr = ecalbegin; ecalitr != firstnotinsc; ++ecalitr ) {    
    const PFClusterElement* elemascluster = 
      docast(const PFClusterElement*,ecalitr->first);
    if( ecalitr->second != false ) {
      ecalclusters.push_back(std::make_pair(elemascluster,true));
      ecalitr->second = false;
    } else {
      std::stringstream ecal_err;
      elemascluster->Dump(ecal_err,"\t");
      LOGDRESSED("PFEGammaAlgo::unwrapSuperCluster()")
	<< "ECAL Cluster matched to SC is already used! "
	<< "This can happen due to the KF-track pre-cleaning."
	<< std::endl << ecal_err << std::endl;
    }
    // process the ES elements
    // auto is a pair<Iterator,bool> here, bool is false when placing fails
    auto emplaceresult = ecal2ps.emplace(elemascluster,
					 ClusterMap::mapped_type());    
    if( !emplaceresult.second ) {
      std::stringstream clus_err;
      elemascluster->Dump(clus_err,"\t");
      throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
	<< "List of pointers to ECAL block elements contains non-unique items!"
	<< " This is very bad!" << std::endl
	<< "cluster ptr = 0x" << std::hex << elemascluster << std::dec 
	<< std::endl << clus_err.str() << std::endl;
    }    
    ClusterMap::mapped_type& eslist = emplaceresult.first->second;    
    if( is_pf_sc ) {
      npfpsclusters += attachPSClusters(thesc,elemascluster,eslist);
    } else {
      npfpsclusters += attachPSClusters(elemascluster,eslist);    
    }
  } // loop over ecal elements

  // check that we found the right number of PF-PS clusters if this is a 
  // PF supercluster, otherwise all bets are off
  if( is_pf_sc && nscpsclusters != npfpsclusters) {
    std::stringstream sc_err;
    thesc->Dump(sc_err,"\t");
    throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
      << "The number of found PF preshower elements (" 
      << npfpsclusters << ") in block is not the same as"
      << " the number of preshower clusters reported by the PFSuperCluster"
      << " itself (" << nscpsclusters << ")! This should not happen!" 
      << std::endl 
      << sc_err.str() << std::endl;
  }

  LOGDRESSED("PFEGammaAlgo")
    << " Unwrapped SC has " << npfclusters << " ECAL sub-clusters"
    << " and " << npfpsclusters << " PreShower layers 1 & 2 clusters!" 
    << std::endl; 
  return true;
}



int PFEGammaAlgo::attachPSClusters(const PFSCElement* thesc,
				      const ClusterElement* ecalclus,
				      ClusterMap::mapped_type& eslist) {  
  if( ecalclus->clusterRef()->layer() == PFLayer::ECAL_BARREL ) return 0;
  SuperClusterRef::key_type sc_key = ecalclus->clusterRef().key();
  edm::Ptr<reco::CaloCluster> clusptr = refToPtr(ecalclus->clusterRef());
  EEtoPSElement ecalkey = std::make_pair(clusptr.key(),clusptr);  
  auto assc_ps = std::equal_range(eetops_->cbegin(),
				  eetops_->cend(),
				  ecalkey,
				  comparePSMapByKey);
  for( const auto& ps1 : _splayedblock[reco::PFBlockElement::PS1] ) {
    edm::Ptr<reco::CaloCluster> temp = refToPtr(ps1.first->clusterRef());
    for( auto pscl = assc_ps.first; pscl != assc_ps.second; ++pscl ) {
      if( pscl->second == temp ) {
	const ClusterElement* pstemp = 
	  docast(const ClusterElement*,ps1.first);
	eslist.push_back( PFClusterFlaggedElement(pstemp,true) );
      }
    }
  }
  for( const auto& ps2 : _splayedblock[reco::PFBlockElement::PS2] ) {
    edm::Ptr<reco::CaloCluster> temp = refToPtr(ps2.first->clusterRef());
    for( auto pscl = assc_ps.first; pscl != assc_ps.second; ++pscl ) {
      if( pscl->second == temp ) {
	const ClusterElement* pstemp = 
	  docast(const ClusterElement*,ps2.first);
	eslist.push_back( PFClusterFlaggedElement(pstemp,true) );
      }
    }
  }
  return eslist.size();
}

int PFEGammaAlgo::attachPSClusters(const ClusterElement* ecalclus,
				      ClusterMap::mapped_type& eslist) {  
  // get PS elements closest to this ECAL cluster and no other
  NotCloserToOther<reco::PFBlockElement::ECAL,reco::PFBlockElement::PS1> 
    ps1ClusterMatch(_currentblock,_currentlinks,ecalclus);
  NotCloserToOther<reco::PFBlockElement::ECAL,reco::PFBlockElement::PS2> 
    ps2ClusterMatch(_currentblock,_currentlinks,ecalclus);
  auto ps1notassc = 
    std::partition(_splayedblock[reco::PFBlockElement::PS1].begin(),
		   _splayedblock[reco::PFBlockElement::PS1].end(),
		   ps1ClusterMatch);
  auto ps2notassc = 
    std::partition(_splayedblock[reco::PFBlockElement::PS2].begin(),
		   _splayedblock[reco::PFBlockElement::PS2].end(),
		   ps2ClusterMatch);
  auto ps1begin = _splayedblock[reco::PFBlockElement::PS1].begin();
  auto ps2begin = _splayedblock[reco::PFBlockElement::PS2].begin();
  
  const double npsclustersforcrystal = ( std::distance(ps1begin,ps1notassc) +
					 std::distance(ps2begin,ps2notassc) );
      
  eslist.resize(npsclustersforcrystal);
  
  UsableElementToPSCluster elemtops;    
  auto lastpsclus = eslist.begin();
  lastpsclus = std::transform(ps1begin,ps1notassc,lastpsclus,elemtops);
  std::transform(ps2begin,ps2notassc,lastpsclus,elemtops);   
  if( npsclustersforcrystal != eslist.size() ) {
    throw cms::Exception("PFEGammaAlgo::attachPSClusters()")
      << "Didn't insert correct number of ps clusters!";
  }
  return eslist.size();
}

void PFEGammaAlgo::dumpCurrentRefinableObjects() const {
#ifdef PFLOW_DEBUG
  edm::LogVerbatim("PFEGammaAlgo") 
    //<< "Dumping current block: " << std::endl << *_currentblock << std::endl
    << "Dumping " << _refinableObjects.size()
    << " refinable objects for this block: " << std::endl;
  for( const auto& ro : _refinableObjects ) {    
    std::stringstream info;
    info << "Refinable Object:" << std::endl;
    if( ro.parentSC ) {
      info << "\tSuperCluster element attached to object:" << std::endl 
	   << '\t';
      ro.parentSC->Dump(info,"\t");
      info << std::endl;      
    } 
    if( ro.electronSeed.isNonnull() ) {
      info << "\tGSF element attached to object:" << std::endl;
      ro.primaryGSFs.front().first->Dump(info,"\t");
      info << std::endl;      
    }
    if( ro.primaryKFs.size() ) {
      info << "\tPrimary KF tracks attached to object: " << std::endl;
      for( const auto& kf : ro.primaryKFs ) {
	kf.first->Dump(info,"\t");
	info << std::endl;
      }
    }
    if( ro.secondaryKFs.size() ) {
      info << "\tSecondary KF tracks attached to object: " << std::endl;
      for( const auto& kf : ro.secondaryKFs ) {
	kf.first->Dump(info,"\t");
	info << std::endl;
      }
    }
    if( ro.brems.size() ) {
      info << "\tBrem tangents attached to object: " << std::endl;
      for( const auto& brem : ro.brems ) {
	brem.first->Dump(info,"\t");
	info << std::endl;
      }
    }
    if( ro.ecalclusters.size() ) {
      info << "\tECAL clusters attached to object: " << std::endl;
      for( const auto& clus : ro.ecalclusters ) {
	clus.first->Dump(info,"\t");
	info << std::endl;
	if( ro.ecal2ps.find(clus.first) != ro.ecal2ps.end() ) {
	  for( const auto& psclus : ro.ecal2ps.at(clus.first) ) {
	    info << "\t\t Attached PS Cluster: ";
	    psclus.first->Dump(info,"");
	    info << std::endl;
	  }
	}
      }
    }
    edm::LogVerbatim("PFEGammaAlgo") << info.str();
  }
#endif
}

// look through our KF tracks in this block and match 
void PFEGammaAlgo::
removeOrLinkECALClustersToKFTracks() {
  if( !_splayedblock[reco::PFBlockElement::ECAL].size() ||
      !_splayedblock[reco::PFBlockElement::TRACK].size()   ) return;
  std::multimap<double, unsigned> matchedGSFs, matchedECALs;
  for( auto& kftrack : _splayedblock[reco::PFBlockElement::TRACK] ) {
    matchedGSFs.clear();
    _currentblock->associatedElements(kftrack.first->index(), _currentlinks,
				      matchedGSFs,
				      reco::PFBlockElement::GSF,
				      reco::PFBlock::LINKTEST_ALL);
    if( !matchedGSFs.size() ) { // only run this is we aren't associated to GSF
      LesserByDistance closestTrackToECAL(_currentblock,_currentlinks,
					  &kftrack);      
      auto ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
      auto ecalend   = _splayedblock[reco::PFBlockElement::ECAL].end();
      std::partial_sort(ecalbegin,ecalbegin+1,ecalend,closestTrackToECAL);
      PFFlaggedElement& closestECAL = 
	_splayedblock[reco::PFBlockElement::ECAL].front();
      const float dist = _currentblock->dist(kftrack.first->index(), 
					     closestECAL.first->index(),
					     _currentlinks,
					     reco::PFBlock::LINKTEST_ALL);
      bool inSC = false;
      for( auto& sc : _splayedblock[reco::PFBlockElement::SC] ) {
	float dist_sc = _currentblock->dist(sc.first->index(), 
					    closestECAL.first->index(),
					    _currentlinks,
					    reco::PFBlock::LINKTEST_ALL);
	if( dist_sc != -1.0f) { inSC = true; break; }
      }
      
      if( dist != -1.0f && closestECAL.second ) {
	bool gsflinked = false;
	// check that this cluster is not associated to a GSF track
	for(const auto& gsfflag : _splayedblock[reco::PFBlockElement::GSF]) {
	  const reco::PFBlockElementGsfTrack* elemasgsf =
	    docast(const reco::PFBlockElementGsfTrack*,gsfflag.first);
	  if(elemasgsf->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) {
	    continue; // keep clusters that have a found conversion GSF near
	  }
	  matchedECALs.clear();
	  _currentblock->associatedElements(elemasgsf->index(), _currentlinks,
					    matchedECALs,
					    reco::PFBlockElement::ECAL,
					    reco::PFBlock::LINKTEST_ALL);
	  if( matchedECALs.size() ) {
	    if( matchedECALs.begin()->second == closestECAL.first->index() ) {
	      gsflinked = true;
	      break;
	    }
	  }			    
	} // loop over primary GSF tracks
	if( !gsflinked && !inSC) { 
	  // determine if we should remove the matched cluster
	  const reco::PFBlockElementTrack * kfEle = 
	    docast(const reco::PFBlockElementTrack*,kftrack.first);
	  const reco::TrackRef trackref = kfEle->trackRef();
	  const unsigned Algo = trackref->algo();
	  const int nexhits = 
	    trackref->trackerExpectedHitsInner().numberOfLostHits();
	  bool fromprimaryvertex = false;
	  for( auto vtxtks = cfg_.primaryVtx->tracks_begin();
	       vtxtks != cfg_.primaryVtx->tracks_end(); ++ vtxtks ) {
	    if( trackref == vtxtks->castTo<reco::TrackRef>() ) {
	      fromprimaryvertex = true;
	      break;
	    }
	  }// loop over tracks in primary vertex
	   // if associated to good non-GSF matched track remove this cluster
	  if( Algo < 9 && nexhits == 0 && fromprimaryvertex ) {
	    closestECAL.second = false;
	  } else { // otherwise associate the cluster and KF track
	    _recoveredlinks.push_back( ElementMap::value_type(closestECAL.first,kftrack.first) );
	    _recoveredlinks.push_back( ElementMap::value_type(kftrack.first,closestECAL.first) );
	  }
	}
      } // found a good closest ECAL match
    } // no GSF track matched to KF
  } // loop over KF elements
}

void PFEGammaAlgo::
mergeROsByAnyLink(std::list<PFEGammaAlgo::ProtoEGObject>& ROs) {
  if( ROs.size() < 2 ) return; // nothing to do with one or zero ROs  
  bool check_for_merge = true;
  while( check_for_merge ) {    
    ProtoEGObject& thefront = ROs.front();
    TestIfROMergableByLink mergeTest(thefront);
    auto mergestart = ROs.begin(); ++mergestart;    
    auto nomerge = std::partition(mergestart,ROs.end(),mergeTest);
    if( nomerge != mergestart ) {
      LOGDRESSED("PFEGammaAlgo::mergeROsByAnyLink()")
	<< "Found objects to merge by links to the front!" << std::endl;
      for( auto roToMerge = mergestart; roToMerge != nomerge; ++roToMerge) {
	thefront.ecalclusters.insert(thefront.ecalclusters.end(),
				     roToMerge->ecalclusters.begin(),
				     roToMerge->ecalclusters.end());
	thefront.ecal2ps.insert(roToMerge->ecal2ps.begin(),
				roToMerge->ecal2ps.end());
	thefront.secondaryKFs.insert(thefront.secondaryKFs.end(),
				     roToMerge->secondaryKFs.begin(),
				     roToMerge->secondaryKFs.end());
	if( !thefront.parentSC && roToMerge->parentSC ) {
	  thefront.parentSC = roToMerge->parentSC;
	}
	if( thefront.electronSeed.isNull() && 
	    roToMerge->electronSeed.isNonnull() ) {
	  thefront.electronSeed = roToMerge->electronSeed;
	  thefront.primaryGSFs.insert(thefront.primaryGSFs.end(),
				      roToMerge->primaryGSFs.begin(),
				      roToMerge->primaryGSFs.end());
	  thefront.primaryKFs.insert(thefront.primaryKFs.end(),
				     roToMerge->primaryKFs.begin(),
				     roToMerge->primaryKFs.end());
	  thefront.brems.insert(thefront.brems.end(),
				roToMerge->brems.begin(),
				roToMerge->brems.end());
	}
      }      
      ROs.erase(mergestart,nomerge);
      // put the merged element in the back of the cleaned list
      ROs.push_back(ROs.front());
      ROs.pop_front();
    } else {      
      check_for_merge = false;
    }
  }
  LOGDRESSED("PFEGammaAlgo::mergeROsByAnyLink()") 
	<< "After merging by links there are: " << ROs.size() 
	<< " refinable EGamma objects!" << std::endl;
}

// pull in KF tracks associated to the RO but not closer to another
// NB: in initializeProtoCands() we forced the GSF tracks not to be 
//     from a conversion, but we will leave a protection here just in
//     case things change in the future
void PFEGammaAlgo::
linkRefinableObjectGSFTracksToKFs(ProtoEGObject& RO) {
  constexpr reco::PFBlockElement::TrackType convType = 
    reco::PFBlockElement::T_FROM_GAMMACONV;
  if( !_splayedblock[reco::PFBlockElement::TRACK].size() ) return;
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  for( auto& gsfflagged : RO.primaryGSFs ) {
    const PFGSFElement* seedtk = gsfflagged.first;
    // don't process SC-only ROs or secondary seeded ROs
    if( RO.electronSeed.isNull() || seedtk->trackType(convType) ) continue;
    NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::TRACK>
      gsfTrackToKFs(_currentblock,_currentlinks,seedtk);
    // get KF tracks not closer to another and not already used
    auto notlinked = std::partition(KFbegin,KFend,gsfTrackToKFs);
    // attach tracks and set as used
    for( auto kft = KFbegin; kft != notlinked; ++kft ) {
      const PFKFElement* elemaskf = 
	docast(const PFKFElement*,kft->first);
      // don't care about things that aren't primaries or directly 
      // associated secondary tracks
      if( isPrimaryTrack(*elemaskf,*seedtk) &&
	  !elemaskf->trackType(convType)       ) {
	kft->second = false;
	RO.primaryKFs.push_back(std::make_pair(elemaskf,true));
	RO.localMap.push_back( ElementMap::value_type(seedtk,elemaskf) );
	RO.localMap.push_back( ElementMap::value_type(elemaskf,seedtk) );
      } else if ( elemaskf->trackType(convType) ) {
	kft->second = false;
	RO.secondaryKFs.push_back(std::make_pair(elemaskf,true));
	RO.localMap.push_back( ElementMap::value_type(seedtk,elemaskf) );
	RO.localMap.push_back( ElementMap::value_type(elemaskf,seedtk) );
      }
    }// loop on closest KFs not closer to other GSFs
  } // loop on GSF primaries on RO  
}

void PFEGammaAlgo::
linkRefinableObjectPrimaryKFsToSecondaryKFs(ProtoEGObject& RO) {
  constexpr reco::PFBlockElement::TrackType convType = 
    reco::PFBlockElement::T_FROM_GAMMACONV;
  if( !_splayedblock[reco::PFBlockElement::TRACK].size() ) return;
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  for( auto& kfflagged : RO.primaryKFs ) {
    const PFKFElement* primkf = kfflagged.first;
    // don't process SC-only ROs or secondary seeded ROs
    if( primkf->trackType(convType) ) {
      throw cms::Exception("PFEGammaAlgo::linkRefinableObjectPrimaryKFsToSecondaryKFs()")
	<< "A KF track from conversion has been assigned as a primary!!"
	<< std::endl;
    }
    NotCloserToOther<reco::PFBlockElement::TRACK,reco::PFBlockElement::TRACK,true>
	kfTrackToKFs(_currentblock,_currentlinks,primkf);
    // get KF tracks not closer to another and not already used
    auto notlinked = std::partition(KFbegin,KFend,kfTrackToKFs);
    // attach tracks and set as used
    for( auto kft = KFbegin; kft != notlinked; ++kft ) {
      const PFKFElement* elemaskf = 
	docast(const PFKFElement*,kft->first);
      // don't care about things that aren't primaries or directly 
      // associated secondary tracks
      if( elemaskf->trackType(convType) ) {
	kft->second = false;
	RO.secondaryKFs.push_back(std::make_pair(elemaskf,true));
	RO.localMap.push_back( ElementMap::value_type(primkf,elemaskf) );
	RO.localMap.push_back( ElementMap::value_type(elemaskf,primkf) );
      } 
    }// loop on closest KFs not closer to other KFs
  } // loop on KF primaries on RO
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::
linkRefinableObjectPrimaryGSFTrackToECAL(ProtoEGObject& RO) {
  if( !_splayedblock[reco::PFBlockElement::ECAL].size() ) return; 
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
  for( auto& primgsf : RO.primaryGSFs ) {
    NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::ECAL>
      gsfTracksToECALs(_currentblock,_currentlinks,primgsf.first);
    CompatibleEoPOut eoverp_test(primgsf.first);
    auto notmatched = std::partition(ECALbegin,ECALend,gsfTracksToECALs);
    notmatched = std::stable_partition(ECALbegin,notmatched,eoverp_test);
    // early termination if there's nothing matched!
    if( !std::distance(ECALbegin,notmatched) ) return;
    // only take the cluster that's closest and well matched
    const PFClusterElement* elemascluster = 
      docast(const PFClusterElement*,ECALbegin->first);    
    PFClusterFlaggedElement temp(elemascluster,true);
    auto cluster = std::find(RO.ecalclusters.begin(),
			     RO.ecalclusters.end(),
			     temp);
    // if the cluster isn't found push it to the front since it 
    // should be the 'seed'
    // if the cluster is already found in the associated clusters
    // then we push it to the front  
    if( cluster == RO.ecalclusters.end() ) {
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") 
	<< "Found a cluster not already associated to GSF extrapolation"
	<< " at ECAL surface!" << std::endl;
      RO.ecalclusters.insert(RO.ecalclusters.begin(),temp);
      attachPSClusters(elemascluster,RO.ecal2ps[elemascluster]);      
      RO.localMap.push_back( ElementMap::value_type(primgsf.first,temp.first) );
      RO.localMap.push_back( ElementMap::value_type(temp.first,primgsf.first) );
      ECALbegin->second = false;
    } else {
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") 
	<< "Found a cluster already associated to GSF extrapolation"
	<< " at ECAL surface!" << std::endl;
      RO.localMap.push_back( ElementMap::value_type(primgsf.first,temp.first) );
      RO.localMap.push_back( ElementMap::value_type(temp.first,primgsf.first) );
      std::swap(*RO.ecalclusters.begin(),*cluster);
    }
  }
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::
linkRefinableObjectKFTracksToECAL(ProtoEGObject& RO) {
  if( !_splayedblock[reco::PFBlockElement::ECAL].size() ) return;  
  for( auto& primkf : RO.primaryKFs ) linkKFTrackToECAL(primkf,RO);
  for( auto& secdkf : RO.secondaryKFs ) linkKFTrackToECAL(secdkf,RO);
}

void PFEGammaAlgo::linkKFTrackToECAL(const KFFlaggedElement& kfflagged,
					ProtoEGObject& RO) {
  std::vector<PFClusterFlaggedElement>& currentECAL = RO.ecalclusters;
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
  NotCloserToOther<reco::PFBlockElement::TRACK,reco::PFBlockElement::ECAL>
    kfTrackToECALs(_currentblock,_currentlinks,kfflagged.first);      
  NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::ECAL>
    kfTrackGSFToECALs(_currentblock,_currentlinks,kfflagged.first);
  //get the ECAL elements not used and not closer to another KF
  auto notmatched = std::partition(ECALbegin,ECALend,kfTrackToECALs);
  //get subset ECAL elements not used or closer to another GSF of any type
  notmatched = std::partition(ECALbegin,notmatched,kfTrackGSFToECALs);
  for( auto ecalitr = ECALbegin; ecalitr != notmatched; ++ecalitr ) {
    const PFClusterElement* elemascluster = 
      docast(const PFClusterElement*,ecalitr->first);
    PFClusterFlaggedElement flaggedclus(elemascluster,true);
    auto existingCluster = std::find(currentECAL.begin(),currentECAL.end(),
				     flaggedclus);
    if( existingCluster == currentECAL.end() ) {
      RO.ecalclusters.push_back(flaggedclus);
      attachPSClusters(elemascluster,RO.ecal2ps[elemascluster]);	  
      ecalitr->second = false;
    }
    RO.localMap.push_back( ElementMap::value_type(elemascluster,kfflagged.first) );
    RO.localMap.push_back( ElementMap::value_type(kfflagged.first,elemascluster) );
  }
}

void PFEGammaAlgo::
linkRefinableObjectBremTangentsToECAL(ProtoEGObject& RO) {
  for( auto& bremflagged : RO.brems ) {
    auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
    auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
    NotCloserToOther<reco::PFBlockElement::BREM,reco::PFBlockElement::ECAL>
      BremToECALs(_currentblock,_currentlinks,bremflagged.first);      
    auto notmatched = std::partition(ECALbegin,ECALend,BremToECALs);
    for( auto ecal = ECALbegin; ecal != notmatched; ++ecal ) {
      float deta = 
	std::abs( ecal->first->clusterRef()->positionREP().eta() -
		  bremflagged.first->positionAtECALEntrance().eta() );
      if( deta < 0.015 ) { 
	const PFClusterElement* elemasclus =
	  docast(const PFClusterElement*,ecal->first);    
	RO.ecalclusters.push_back(std::make_pair(elemasclus,true));
	attachPSClusters(elemasclus,RO.ecal2ps[elemasclus]);
	RO.localMap.push_back( ElementMap::value_type(ecal->first,bremflagged.first) );
	RO.localMap.push_back( ElementMap::value_type(bremflagged.first,ecal->first) );
	LOGDRESSED("PFEGammaAlgo::linkBremToECAL()") 
	  << "Found a cluster not already associated by brem extrapolation"
	  << " at ECAL surface!" << std::endl;
      }
    }
  }  
}

void PFEGammaAlgo::
linkRefinableObjectConvSecondaryKFsToSecondaryKFs(ProtoEGObject& RO) {
  IsConversionTrack<reco::PFBlockElementTrack> isConvKf; 
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend   = _splayedblock[reco::PFBlockElement::TRACK].end();
  auto BeginROskfs = RO.secondaryKFs.begin();
  auto EndROskfs   = RO.secondaryKFs.end();  
  auto ronotconv = std::partition(BeginROskfs,EndROskfs,isConvKf); 
  size_t convkfs_end = std::distance(BeginROskfs,ronotconv);  
  for( size_t idx = 0; idx < convkfs_end; ++idx ) { 
    const PFKFFlaggedElement& ro_skf = RO.secondaryKFs[idx];
    NotCloserToOther<reco::PFBlockElement::TRACK,
                     reco::PFBlockElement::TRACK,
                     true>
      TracksToTracks(_currentblock,_currentlinks, ro_skf.first); 
    auto notmatched = std::partition(KFbegin,KFend,TracksToTracks);    
    notmatched = std::partition(KFbegin,notmatched,isConvKf);    
    for( auto kf = KFbegin; kf != notmatched; ++kf ) {
      const reco::PFBlockElementTrack* elemaskf =
	docast(const reco::PFBlockElementTrack*,kf->first);      
      RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
      RO.localMap.push_back( ElementMap::value_type(ro_skf.first,kf->first) );
      RO.localMap.push_back( ElementMap::value_type(kf->first,ro_skf.first) );
      kf->second = false;      
    }    
  }
}

void PFEGammaAlgo::
linkRefinableObjectECALToSingleLegConv(ProtoEGObject& RO) { 
  IsConversionTrack<reco::PFBlockElementTrack> isConvKf;
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();  
  for( auto& ecal : RO.ecalclusters ) {
    NotCloserToOther<reco::PFBlockElement::ECAL,
                     reco::PFBlockElement::TRACK,
                     true>
      ECALToTracks(_currentblock,_currentlinks,ecal.first);           
    auto notmatchedkf  = std::partition(KFbegin,KFend,ECALToTracks);
    auto notconvkf     = std::partition(KFbegin,notmatchedkf,isConvKf);    
    // add identified KF conversion tracks
    for( auto kf = KFbegin; kf != notconvkf; ++kf ) {
      const reco::PFBlockElementTrack* elemaskf =
	docast(const reco::PFBlockElementTrack*,kf->first);
      RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
      RO.localMap.push_back( ElementMap::value_type(ecal.first,elemaskf) );
      RO.localMap.push_back( ElementMap::value_type(elemaskf,ecal.first) );
      kf->second = false;
    }
    // go through non-conv-identified kfs and check MVA to add conversions
    for( auto kf = notconvkf; kf != notmatchedkf; ++kf ) {
      if( EvaluateSingleLegMVA(_currentblock, *cfg_.primaryVtx, 
			       kf->first->index()) ) {
	const reco::PFBlockElementTrack* elemaskf =
	  docast(const reco::PFBlockElementTrack*,kf->first);
	RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
	RO.localMap.push_back( ElementMap::value_type(ecal.first,elemaskf) );
	RO.localMap.push_back( ElementMap::value_type(elemaskf,ecal.first) );
	kf->second = false;
      }
    }    
  }
}

void  PFEGammaAlgo::
linkRefinableObjectSecondaryKFsToECAL(ProtoEGObject& RO) {
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end(); 
  for( auto& skf : RO.secondaryKFs ) {
    NotCloserToOther<reco::PFBlockElement::TRACK,
                     reco::PFBlockElement::ECAL,
                     false>
      TracksToECALwithCut(_currentblock,_currentlinks,skf.first,1.5f);
    auto notmatched = std::partition(ECALbegin,ECALend,TracksToECALwithCut);
    for( auto ecal = ECALbegin; ecal != notmatched; ++ecal ) {
      const reco::PFBlockElementCluster* elemascluster =
	docast(const reco::PFBlockElementCluster*,ecal->first);      
      RO.ecalclusters.push_back(std::make_pair(elemascluster,true));
      attachPSClusters(elemascluster,RO.ecal2ps[elemascluster]);
      RO.localMap.push_back(ElementMap::value_type(skf.first,elemascluster));
      RO.localMap.push_back(ElementMap::value_type(elemascluster,skf.first));
      ecal->second = false;      
    }
  }
}

void PFEGammaAlgo::
fillPFCandidates(const std::list<PFEGammaAlgo::ProtoEGObject>& ROs,
		 reco::PFCandidateCollection& egcands,
		 reco::PFCandidateEGammaExtraCollection& egxs) {
  // reset output collections
  egcands.clear();
  egxs.clear();  
  refinedscs_.clear();
  egcands.reserve(ROs.size());
  egxs.reserve(ROs.size());
  refinedscs_.reserve(ROs.size());
  for( const auto& RO : ROs ) {    
    if( RO.ecalclusters.size() == 0  && 
	!cfg_.produceEGCandsWithNoSuperCluster ) continue;
    
    reco::PFCandidate cand;
    reco::PFCandidateEGammaExtra xtra;
    if( RO.primaryGSFs.size() || RO.primaryKFs.size() ) {
      cand.setPdgId(-11); // anything with a primary track is an electron
    } else {
      cand.setPdgId(22); // anything with no primary track is a photon
    }    
    if( RO.primaryKFs.size() ) {
      cand.setCharge(RO.primaryKFs[0].first->trackRef()->charge());
      xtra.setKfTrackRef(RO.primaryKFs[0].first->trackRef());
      cand.setTrackRef(RO.primaryKFs[0].first->trackRef());
      cand.addElementInBlock(_currentblock,RO.primaryKFs[0].first->index());
    }
    if( RO.primaryGSFs.size() ) {        
      cand.setCharge(RO.primaryGSFs[0].first->GsftrackRef()->chargeMode());
      xtra.setGsfTrackRef(RO.primaryGSFs[0].first->GsftrackRef());
      cand.setGsfTrackRef(RO.primaryGSFs[0].first->GsftrackRef());
      cand.addElementInBlock(_currentblock,RO.primaryGSFs[0].first->index());
    }
    if( RO.parentSC ) {
      xtra.setSuperClusterBoxRef(RO.parentSC->superClusterRef());      
      // we'll set to the refined supercluster back up in the producer
      cand.setSuperClusterRef(RO.parentSC->superClusterRef());
      xtra.setSuperClusterRef(RO.parentSC->superClusterRef());      
      cand.addElementInBlock(_currentblock,RO.parentSC->index());
    }
    // add brems
    for( const auto& bremflagged : RO.brems ) {
      const PFBremElement* brem = bremflagged.first;
      cand.addElementInBlock(_currentblock,brem->index());      
    }
    // add clusters and ps clusters
    for( const auto& ecal : RO.ecalclusters ) {
      const PFClusterElement* clus = ecal.first;
      cand.addElementInBlock(_currentblock,clus->index());      
      for( auto& ps : RO.ecal2ps.at(clus) ) {
	const PFClusterElement* psclus = ps.first;
	cand.addElementInBlock(_currentblock,psclus->index());	
      }
    }
    // add secondary tracks
    for( const auto& secdkf : RO.secondaryKFs ) {
      const PFKFElement* kf = secdkf.first;
      cand.addElementInBlock(_currentblock,kf->index());
      reco::ConversionRef convref = kf->convRef();
      if( convref.isNonnull() && convref.isAvailable() ) {
	xtra.addConversionRef(convref);
      } else {
	xtra.addSingleLegConvTrackRef(kf->trackRef());
	// just hack it for now FIXME
	xtra.addSingleLegConvMva(-999.9f); 
      }
    }

    // build the refined supercluster from those clusters left in the cand
    refinedscs_.push_back(buildRefinedSuperCluster(RO));
    
    const reco::SuperCluster& the_sc = refinedscs_.back();
    // with the refined SC in hand we build a naive candidate p4 
    // and set the candidate ECAL position to either the barycenter of the 
    // supercluster (if super-cluster present) or the seed of the
    // new SC generated by the EGAlgo 
    const double scE = the_sc.energy();
    if( scE != 0.0 ) {
      const math::XYZPoint& seedPos = the_sc.seed()->position();
      math::XYZVector egDir = the_sc.position()-cfg_.primaryVtx->position();
      egDir = egDir.Unit();      
      cand.setP4(math::XYZTLorentzVector(scE*egDir.x(),
					 scE*egDir.y(),
					 scE*egDir.z(),
					 scE           ));
      math::XYZPointF ecalPOS_f(seedPos.x(),seedPos.y(),seedPos.z());
      cand.setPositionAtECALEntrance(ecalPOS_f);
    } else if ( cfg_.produceEGCandsWithNoSuperCluster && 
		RO.primaryGSFs.size() ) {
      const PFGSFElement* gsf = RO.primaryGSFs[0].first;
      reco::GsfTrackRef gref = gsf->GsftrackRef();
      math::XYZTLorentzVector p4(gref->pxMode(),gref->pyMode(),
				 gref->pzMode(),gref->pMode());
      cand.setP4(p4);      
      cand.setPositionAtECALEntrance(gsf->positionAtECALEntrance());
    } else if ( cfg_.produceEGCandsWithNoSuperCluster &&
		RO.primaryKFs.size() ) {
      const PFKFElement* kf = RO.primaryKFs[0].first;
      reco::TrackRef kref = RO.primaryKFs[0].first->trackRef();
      math::XYZTLorentzVector p4(kref->px(),kref->py(),kref->pz(),kref->p());
      cand.setP4(p4);   
      cand.setPositionAtECALEntrance(kf->positionAtECALEntrance());
    }    
    egcands.push_back(cand);
    egxs.push_back(xtra);    
  }
}

// currently stolen from PFECALSuperClusterAlgo, we should
// try to factor this correctly since the operation is the same in
// both places...
reco::SuperCluster PFEGammaAlgo::
buildRefinedSuperCluster(const PFEGammaAlgo::ProtoEGObject& RO) {
  if( !RO.ecalclusters.size() ) { 
    return reco::SuperCluster(0.0,math::XYZPoint(0,0,0));
  }
	
  SumPSEnergy sumps1(reco::PFBlockElement::PS1), 
    sumps2(reco::PFBlockElement::PS2);  
						  
  bool isEE = false;
  edm::Ptr<reco::PFCluster> clusptr;
  // need the vector of raw pointers for a PF width class
  std::vector<const reco::PFCluster*> bare_ptrs;
  // calculate necessary parameters and build the SC
  double posX(0), posY(0), posZ(0),
    rawSCEnergy(0), corrSCEnergy(0), corrPSEnergy(0),
    PS1_clus_sum(0), PS2_clus_sum(0);  
  for( auto& clus : RO.ecalclusters ) {
    isEE = PFLayer::ECAL_ENDCAP == clus.first->clusterRef()->layer();
    clusptr = 
      edm::refToPtr<reco::PFClusterCollection>(clus.first->clusterRef());
    bare_ptrs.push_back(clusptr.get());    

    const double cluseraw = clusptr->energy();
    const double cluscalibe_nops = 
      cfg_.thePFEnergyCalibration->energyEm(*clusptr,
					    0.0,0.0,
					    false);
    double cluscalibe_ps = cluscalibe_nops;
    const math::XYZPoint& cluspos = clusptr->position();
    posX += cluseraw * cluspos.X();
    posY += cluseraw * cluspos.Y();
    posZ += cluseraw * cluspos.Z();
    // update EE calibrated super cluster energies
    if( isEE ) {
      const auto& psclusters = RO.ecal2ps.at(clus.first);
      PS1_clus_sum = std::accumulate(psclusters.begin(),psclusters.end(),
				     0.0,sumps1);
      PS2_clus_sum = std::accumulate(psclusters.begin(),psclusters.end(),
				     0.0,sumps2);
      cluscalibe_ps = 
	cfg_.thePFEnergyCalibration->energyEm(*clusptr,
					      PS1_clus_sum,PS2_clus_sum,
					      cfg_.applyCrackCorrections);
    }

    rawSCEnergy  += cluseraw;
    corrSCEnergy += cluscalibe_ps;    
    corrPSEnergy += cluscalibe_ps - cluscalibe_nops;    
  }
  posX /= rawSCEnergy;
  posY /= rawSCEnergy;
  posZ /= rawSCEnergy;

  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy,math::XYZPoint(posX,posY,posZ)); 
  double ps1_energy(0.0), ps2_energy(0.0);
  clusptr = 
    edm::refToPtr<reco::PFClusterCollection>(RO.ecalclusters.front().
					     first->clusterRef());
  new_sc.setSeed(clusptr);
  new_sc.setPreshowerEnergy(corrPSEnergy); 
  for( const auto& clus : RO.ecalclusters ) {
    clusptr = 
      edm::refToPtr<reco::PFClusterCollection>(clus.first->clusterRef());
    new_sc.addCluster(clusptr);
    auto& hits_and_fractions = clusptr->hitsAndFractions();
    for( auto& hit_and_fraction : hits_and_fractions ) {
      new_sc.addHitAndFraction(hit_and_fraction.first,hit_and_fraction.second);
    }
     // put the preshower stuff back in later
    const auto& cluspsassociation = RO.ecal2ps.at(clus.first);
    // EE rechits should be uniquely matched to sets of pre-shower
    // clusters at this point, so we throw an exception if otherwise
    // now wrapped in EDM debug flags
    for( const auto& pscluselem : cluspsassociation ) {    
      edm::Ptr<reco::PFCluster> psclus = 
	  edm::refToPtr<reco::PFClusterCollection>(pscluselem.first->
						   clusterRef());
#ifdef PFFLOW_DEBUG
      auto found_pscluster = std::find(new_sc.preshowerClustersBegin(),
				       new_sc.preshowerClustersEnd(),
				       reco::CaloClusterPtr(psclus));
      if( found_pscluster == new_sc.preshowerClustersEnd() ) {
#endif		  
	const double psenergy = psclus->energy();
	const PFLayer::Layer pslayer = psclus->layer();
	new_sc.addPreshowerCluster(psclus);
	ps1_energy += (PFLayer::PS1 == pslayer)*psenergy;
	ps2_energy += (PFLayer::PS2 == pslayer)*psenergy;
#ifdef PFFLOW_DEBUG
      } else {
	throw cms::Exception("PFECALSuperClusterAlgo::buildSuperCluster")
	  << "Found a PS cluster matched to more than one EE cluster!" 
	  << std::endl << std::hex << psclus.get() << " == " 
	  << found_pscluster->get() << std::dec << std::endl;
      }
#endif
    }    
  }
  new_sc.setPreshowerEnergyPlane1(ps1_energy);
  new_sc.setPreshowerEnergyPlane2(ps2_energy);
  
  // calculate linearly weighted cluster widths
  PFClusterWidthAlgo pfwidth(bare_ptrs);
  new_sc.setEtaWidth(pfwidth.pflowEtaWidth());
  new_sc.setPhiWidth(pfwidth.pflowPhiWidth());
  
  // cache the value of the raw energy  
  new_sc.rawEnergy();
  
  return new_sc;
}

void PFEGammaAlgo::
unlinkRefinableObjectKFandECALWithBadEoverP(ProtoEGObject& RO) {  
  // this only means something for ROs with a primary GSF track
  if( !RO.primaryGSFs.size() ) return;  
  // need energy sums to tell if we've added crap or not
  const double Pin_gsf = RO.primaryGSFs.front().first->GsftrackRef()->pMode();
  const double gsfOuterEta  = 
    RO.primaryGSFs.front().first->positionAtECALEntrance().Eta();
  double tot_ecal= 0.0;  
  std::vector<double> min_brem_dists;
  std::vector<double> closest_brem_eta;
  std::vector<PFKFFlaggedElement> kfs_to_remove;
  std::vector<PFClusterFlaggedElement> ecals_to_remove;
  auto seckfs_begin = RO.secondaryKFs.begin();
  auto seckfs_end   = RO.secondaryKFs.end();
  auto ecal_begin = RO.ecalclusters.begin();
  auto ecal_end   = RO.ecalclusters.end();
  // first get the total ecal energy (we should replace this with a cache)
  for( const auto& ecal : RO.ecalclusters ) {
    tot_ecal += ecal.first->clusterRef()->energy();
    // we also need to look at the minimum distance to brems
    // since energetic brems will be closer to the brem than the track
    double min_brem_dist = 5000.0; 
    double eta = -999.0;
    for( const auto& brem : RO.brems ) {
      const float dist = _currentblock->dist(brem.first->index(),
					     ecal.first->index(),
					     _currentlinks,
					     reco::PFBlock::LINKTEST_ALL);
      if( dist < min_brem_dist && dist != -1.0f ) {
	min_brem_dist = dist;
	eta = brem.first->positionAtECALEntrance().Eta();
      }
    }
    min_brem_dists.push_back(min_brem_dist);
    closest_brem_eta.push_back(eta);
  }  
  
  // loop through the ECAL clusters and remove ECAL clusters matched to
  // secondary track either in *or* out of the SC if the E/pin is bad
  for( auto secd_kf = seckfs_begin; secd_kf != seckfs_end; ++secd_kf ) {
    reco::TrackRef trkRef =   secd_kf->first->trackRef();   
    const float secpin = secd_kf->first->trackRef()->p();       
    for( auto ecal = ecal_begin; ecal != ecal_end; ++ecal ) {
      const float minbremdist = min_brem_dists[std::distance(ecal_begin,ecal)];
      const double ecalenergy = ecal->first->clusterRef()->energy();
      const double Epin = ecalenergy/secpin;
      const double detaGsf = 
	std::abs(gsfOuterEta - ecal->first->clusterRef()->positionREP().Eta());
      const double detaBrem = 
	std::abs(closest_brem_eta[std::distance(ecal_begin,ecal)] - 
		 ecal->first->clusterRef()->positionREP().Eta());
      
      ElementMap::value_type check_match(ecal->first,secd_kf->first);
      auto kf_matched = std::find(RO.localMap.begin(),
				  RO.localMap.end(),
				  check_match);
      
      const float tkdist = _currentblock->dist(secd_kf->first->index(),
					       ecal->first->index(),
					       _currentlinks,
					       reco::PFBlock::LINKTEST_ALL);
      
      // do not reject this track if it is closer to a brem than the
      // secondary track, or if it lies in the delta-eta plane with the
      // gsf track or if it is in the dEta plane with the brems
      if( Epin > 3 && kf_matched != RO.localMap.end() && 
	  tkdist != -1.0f && tkdist < minbremdist &&
	  detaGsf > 0.05 && detaBrem > 0.015) {
	double res_with = std::abs((tot_ecal-Pin_gsf)/Pin_gsf);
	double res_without = std::abs((tot_ecal-ecalenergy-Pin_gsf)/Pin_gsf);
	if(res_without < res_with) {	  
	    LOGDRESSED("PFEGammaAlgo")
	      << " REJECTED_RES totenergy " << tot_ecal
	      << " Pin_gsf " << Pin_gsf 
	      << " cluster to secondary " <<  ecalenergy
	      << " res_with " <<  res_with
	      << " res_without " << res_without << std::endl;
	    tot_ecal -= ecalenergy;
	    ecals_to_remove.push_back(*ecal);
	    kfs_to_remove.push_back(*secd_kf);
	}
      }
    }
  }  
  for( auto& to_remove : ecals_to_remove ) {
    std::remove(RO.ecalclusters.begin(),RO.ecalclusters.end(),to_remove);
  }  
  for( auto& to_remove : kfs_to_remove ) {
    std::remove(RO.secondaryKFs.begin(),RO.secondaryKFs.end(),to_remove);
  }
}

void PFEGammaAlgo::
unlinkRefinableObjectKFandECALMatchedToHCAL(ProtoEGObject& RO,
					    bool removeFreeECAL,
					    bool removeSCEcal) {
  std::vector<bool> cluster_in_sc;
  std::vector<PFKFFlaggedElement> kfs_to_remove;
  auto seckfs_begin = RO.secondaryKFs.begin();
  auto seckfs_end   = RO.secondaryKFs.end();
  auto ecal_begin = RO.ecalclusters.begin();
  auto ecal_end   = RO.ecalclusters.end();
  auto hcal_begin = _splayedblock[reco::PFBlockElement::HCAL].begin();
  auto hcal_end   = _splayedblock[reco::PFBlockElement::HCAL].end();
  for( auto secd_kf = seckfs_begin; secd_kf != seckfs_end; ++secd_kf ) {
    NotCloserToOther<reco::PFBlockElement::TRACK,reco::PFBlockElement::HCAL>
      tracksToHCALs(_currentblock,_currentlinks,secd_kf->first);
    reco::TrackRef trkRef =   secd_kf->first->trackRef();
    const unsigned int Algo = whichTrackAlgo(trkRef);
    const float secpin = trkRef->p();       
    
    for( auto ecal = ecal_begin; ecal != ecal_end; ++ecal ) {
      const double ecalenergy = ecal->first->clusterRef()->energy();
      // first check if the cluster is in the SC (use dist calc for fastness)
      const size_t clus_idx = std::distance(ecal_begin,ecal);
      if( cluster_in_sc.size() < clus_idx + 1) {
	float dist = -1.0f;
	if( RO.parentSC ) {
	  dist = _currentblock->dist(secd_kf->first->index(),
				     ecal->first->index(),
				     _currentlinks,
				     reco::PFBlock::LINKTEST_ALL);
	} 
	cluster_in_sc.push_back(dist != -1.0f); 
      }

      ElementMap::value_type check_match(ecal->first,secd_kf->first);
      auto kf_matched = std::find(RO.localMap.begin(),
				  RO.localMap.end(),
				  check_match);
      // if we've found a secondary KF that matches this ecal cluster
      // now we see if it is matched to HCAL 
      // if it is matched to an HCAL cluster we take different 
      // actions if the cluster was in an SC or not
      if( kf_matched != RO.localMap.end() ) {
	auto hcal_matched = std::partition(hcal_begin,hcal_end,tracksToHCALs);
	for( auto hcalclus = hcal_begin; 
	     hcalclus != hcal_matched; 
	     ++hcalclus                 ) {
	  const reco::PFBlockElementCluster * clusthcal =  
	    dynamic_cast<const reco::PFBlockElementCluster*>(hcalclus->first); 
	  const double hcalenergy = clusthcal->clusterRef()->energy();	  
	  const double hpluse = ecalenergy+hcalenergy;
	  const bool isHoHE = ( (hcalenergy / hpluse ) > 0.1 && Algo < 3 );
	  const bool isHoE  = ( hcalenergy > ecalenergy );
	  const bool isPoHE = ( secpin > hpluse );	
	  if( cluster_in_sc[clus_idx] ) {
	    if(isHoE || isPoHE) {
	      LOGDRESSED("PFEGammaAlgo")
		<< "REJECTED TRACK FOR H/E or P/(H+E), CLUSTER IN SC"
		<< " H/H+E " << (hcalenergy / hpluse)
		<< " H/E " << (hcalenergy > ecalenergy)
		<< " P/(H+E) " << (secpin/hpluse)
		<< " HCAL ENE " << hcalenergy
		<< " ECAL ENE " << ecalenergy
		<< " secPIN " << secpin 
		<< " Algo Track " << Algo << std::endl;
	      kfs_to_remove.push_back(*secd_kf);
	    }
	  } else {
	    if(isHoHE){
	      LOGDRESSED("PFEGammaAlgo")
		<< "REJECTED TRACK FOR H/H+E, CLUSTER NOT IN SC"
		<< " H/H+E " << (hcalenergy / hpluse)
		<< " H/E " << (hcalenergy > ecalenergy)
		<< " P/(H+E) " << (secpin/hpluse) 
		<< " HCAL ENE " << hcalenergy
		<< " ECAL ENE " << ecalenergy
		<< " secPIN " << secpin 
		<< " Algo Track " << Algo << std::endl;
	      kfs_to_remove.push_back(*secd_kf);
	    }
	  }  
	}
      }
    }
  }
  for( auto& to_remove : kfs_to_remove ) {    
    std::remove(RO.secondaryKFs.begin(),RO.secondaryKFs.end(),to_remove);
  }
}


unsigned int PFEGammaAlgo::whichTrackAlgo(const reco::TrackRef& trackRef) {
  unsigned int Algo = 0; 
  switch (trackRef->algo()) {
  case TrackBase::ctf:
  case TrackBase::iter0:
  case TrackBase::iter1:
  case TrackBase::iter2:
    Algo = 0;
    break;
  case TrackBase::iter3:
    Algo = 1;
    break;
  case TrackBase::iter4:
    Algo = 2;
    break;
  case TrackBase::iter5:
    Algo = 3;
    break;
  case TrackBase::iter6:
    Algo = 4;
    break;
  default:
    Algo = 5;
    break;
  }
  return Algo;
}
bool PFEGammaAlgo::isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
				    const reco::PFBlockElementGsfTrack& GsfEl) {
  bool isPrimary = false;
  
  GsfPFRecTrackRef gsfPfRef = GsfEl.GsftrackRefPF();
  
  if(gsfPfRef.isNonnull()) {
    PFRecTrackRef  kfPfRef = KfEl.trackRefPF();
    PFRecTrackRef  kfPfRef_fromGsf = (*gsfPfRef).kfPFRecTrackRef();
    if(kfPfRef.isNonnull() && kfPfRef_fromGsf.isNonnull()) {
      reco::TrackRef kfref= (*kfPfRef).trackRef();
      reco::TrackRef kfref_fromGsf = (*kfPfRef_fromGsf).trackRef();
      if(kfref.isNonnull() && kfref_fromGsf.isNonnull()) {
	if(kfref ==  kfref_fromGsf)
	  isPrimary = true;
      }
    }
  }

  return isPrimary;
}

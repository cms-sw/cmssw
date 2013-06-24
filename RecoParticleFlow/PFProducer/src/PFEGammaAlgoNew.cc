#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgoNew.h"
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
#define PFLOW_DEBUG

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
  typedef PFEGammaAlgoNew::PFSCElement SCElement;
  typedef PFEGammaAlgoNew::EEtoPSAssociation EEtoPSAssociation;
  typedef std::pair<CaloClusterPtr::key_type,CaloClusterPtr> EEtoPSElement;
  typedef PFEGammaAlgoNew::PFClusterElement ClusterElement;
  typedef PFEGammaAlgoNew::PFFlaggedElement PFFlaggedElement;
  typedef PFEGammaAlgoNew::PFSCFlaggedElement SCFlaggedElement;
  typedef PFEGammaAlgoNew::PFKFFlaggedElement KFFlaggedElement;
  typedef PFEGammaAlgoNew::PFGSFFlaggedElement GSFFlaggedElement;
  typedef PFEGammaAlgoNew::PFClusterFlaggedElement ClusterFlaggedElement;

  typedef std::unary_function<const ClusterFlaggedElement&,
			      bool> ClusterMatcher;  
  
  typedef std::unary_function<const PFFlaggedElement&,
			      bool> PFFlaggedElementMatcher; 
  typedef std::binary_function<const PFFlaggedElement&,
			       const PFFlaggedElement&,
			       bool> PFFlaggedElementSorter; 
  
  typedef std::unary_function<const reco::PFBlockElement&,
			      bool> PFElementMatcher; 

  typedef std::unary_function<const PFEGammaAlgoNew::ProtoEGObject&,
			      bool> POMatcher; 
  
  typedef std::unary_function<PFFlaggedElement&, 
			      ClusterFlaggedElement> ClusterElementConverter;

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
    bool operator() (const PFEGammaAlgoNew::ProtoEGObject& po) {
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

    std::multimap<double, unsigned> dists_to_val; 
    const double dist = 
      block->dist(key,test,block->linkData(),reco::PFBlock::LINKTEST_ALL);
    if( dist == -1.0 ) return false; // don't associate non-linked elems
    block->associatedElements(test,block->linkData(),dists_to_val,keytype,
			      reco::PFBlock::LINKTEST_ALL);   
    for( auto& valdist : dists_to_val ) {
      const size_t idx = valdist.second;
      // check track types for conversion info
      switch( keytype ) {
      case reco::PFBlockElement::GSF:
	{
	  const reco::PFBlockElementGsfTrack* elemasgsf  =  
	    docast(const reco::PFBlockElementGsfTrack*,
		   &(block->elements()[idx]));
	  if( !useConvs && elemasgsf->trackType(ConvType) ) continue;
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
	  if( !useConvs && elemaskf->trackType(ConvType) ) continue;
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
						EoPin_cut(EoPcut) {}
    NotCloserToOther(const reco::PFBlockRef& b,
		     const reco::PFBlock::LinkData& l,
		     const reco::PFBlockElement* e,
		     const float EoPcut=1.0e6): comp(e), 
						block(b), 
						links(l),
						EoPin_cut(EoPcut) {}
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
  
  bool isROLinkedByClusterOrTrack(const PFEGammaAlgoNew::ProtoEGObject& RO1,
				  const PFEGammaAlgoNew::ProtoEGObject& RO2 ) {
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
    const PFEGammaAlgoNew::ProtoEGObject& comp;
    TestIfROMergableByLink(const PFEGammaAlgoNew::ProtoEGObject& RO) :
      comp(RO) {}
    bool operator() (const PFEGammaAlgoNew::ProtoEGObject& ro) {      
      return ( isROLinkedByClusterOrTrack(comp,ro) || 
	       isROLinkedByClusterOrTrack(ro,comp)   );      
    }
  }; 
}

PFEGammaAlgoNew::
PFEGammaAlgoNew(const double mvaEleCut,
		std::string  mvaWeightFileEleID,
		const std::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
		const std::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
		bool applyCrackCorrections,
		bool usePFSCEleCalib,
		bool useEGElectrons,
		bool useEGammaSupercluster,
		double sumEtEcalIsoForEgammaSC_barrel,
		double sumEtEcalIsoForEgammaSC_endcap,
		double coneEcalIsoForEgammaSC,
		double sumPtTrackIsoForEgammaSC_barrel,
		double sumPtTrackIsoForEgammaSC_endcap,
		unsigned int nTrackIsoForEgammaSC,
		double coneTrackIsoForEgammaSC,
		std::string mvaweightfile,  
		double mvaConvCut, 
		bool useReg, 
		std::string X0_Map,
		const reco::Vertex& primary,
		double sumPtTrackIsoForPhoton,
		double sumPtTrackIsoSlopeForPhoton
		) : 
  mvaEleCut_(mvaEleCut),
  thePFSCEnergyCalibration_(thePFSCEnergyCalibration),
  thePFEnergyCalibration_(thePFEnergyCalibration),
  applyCrackCorrections_(applyCrackCorrections),
  usePFSCEleCalib_(usePFSCEleCalib),
  useEGElectrons_(useEGElectrons),
  useEGammaSupercluster_(useEGammaSupercluster),
  sumEtEcalIsoForEgammaSC_barrel_(sumEtEcalIsoForEgammaSC_barrel),
  sumEtEcalIsoForEgammaSC_endcap_(sumEtEcalIsoForEgammaSC_endcap),
  coneEcalIsoForEgammaSC_(coneEcalIsoForEgammaSC),
  sumPtTrackIsoForEgammaSC_barrel_(sumPtTrackIsoForEgammaSC_barrel),
  sumPtTrackIsoForEgammaSC_endcap_(sumPtTrackIsoForEgammaSC_endcap),
  nTrackIsoForEgammaSC_(nTrackIsoForEgammaSC),
  coneTrackIsoForEgammaSC_(coneTrackIsoForEgammaSC),  
  isvalid_(false), 
  verbosityLevel_(Silent), 
  MVACUT(mvaConvCut),
  useReg_(useReg),
  sumPtTrackIsoForPhoton_(sumPtTrackIsoForPhoton),
  sumPtTrackIsoSlopeForPhoton_(sumPtTrackIsoSlopeForPhoton),
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
  tmvaReaderEle_->BookMVA("BDT",mvaWeightFileEleID.c_str());
  
  
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
    tmvaReader_->BookMVA("BDT",mvaweightfile.c_str());  

    //Material Map
    TFile *XO_File = new TFile(X0_Map.c_str(),"READ");
    X0_sum=(TH2D*)XO_File->Get("TrackerSum");
    X0_inner = (TH2D*)XO_File->Get("Inner");
    X0_middle = (TH2D*)XO_File->Get("Middle");
    X0_outer = (TH2D*)XO_File->Get("Outer");
    
}

void PFEGammaAlgoNew::RunPFEG(const reco::PFBlockRef&  blockRef,
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

float PFEGammaAlgoNew::EvaluateResMVA(reco::PFCandidate photon, std::vector<reco::CaloCluster>PFClusters){
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

float PFEGammaAlgoNew::EvaluateGCorrMVA(reco::PFCandidate photon, std::vector<CaloCluster>PFClusters){
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

double PFEGammaAlgoNew::ClustersPhiRMS(std::vector<reco::CaloCluster>PFClusters, float PFPhoPhi){
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

float PFEGammaAlgoNew::EvaluateLCorrMVA(reco::PFClusterRef clusterRef ){
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
  VtxZ_=primaryVertex_->z();
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

bool PFEGammaAlgoNew::EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, const reco::Vertex& primaryvtx, unsigned int track_index)  
{  
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
  if(mvaValue > MVACUT)convtkfound=true;  
  return convtkfound;  
}

//Recover Early Conversions reconstructed as PFelectrons
void PFEGammaAlgoNew::EarlyConversion(    
				   //std::auto_ptr< reco::PFCandidateCollection > 
				   //&pfElectronCandidates_,
				   std::vector<reco::PFCandidate>& 
				   tempElectronCandidates,
				   const reco::PFBlockElementSuperCluster* sc
				   ){
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

bool PFEGammaAlgoNew::isAMuon(const reco::PFBlockElement& pfbe) {
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

void PFEGammaAlgoNew::buildAndRefineEGObjects(const reco::PFBlockRef& block) {
  LOGVERB("PFEGammaAlgo") 
    << "Resetting PFEGammaAlgo for new block and running!" << std::endl;
  _splayedblock.clear();
  _recoveredlinks.clear();
  _refinableObjects.clear();
  _finalCandidates.clear();  
  _splayedblock.resize(12); // make sure that we always have the SC ent ry

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
    << " after the linking step." << std::endl;
  dumpCurrentRefinableObjects();

}

void PFEGammaAlgoNew::
initializeProtoCands(std::list<PFEGammaAlgoNew::ProtoEGObject>& egobjs) {
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
    unwrapSuperCluster(fromSC.parentSC,fromSC.ecalclusters,fromSC.ecal2ps);
    _refinableObjects.push_back(fromSC);
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
	fromGSF.localMap.emplace(eAsBrem,elementAsGSF);
	fromGSF.localMap.emplace(elementAsGSF,eAsBrem);
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
	LOGERR("PFEGammaAlgoNew")
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

void PFEGammaAlgoNew::
unwrapSuperCluster(const PFSCElement* thesc,
		   std::list<PFClusterFlaggedElement>& ecalclusters,
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
    throw cms::Exception("PFEGammaAlgoNew::unwrapSuperCluster()")
      << "There are no ECAL elements in a block with imported SC!" 
      << std::endl;
  }
  reco::SuperClusterRef scref = thesc->superClusterRef();
  // this check needs to be done in a different way
  const bool is_pf_sc = (bool)
    docast(const reco::PFCluster*,(*scref->clustersBegin()).get());
  if( !(scref.isAvailable() && scref.isNonnull()) ) {
    throw cms::Exception("PFEGammaAlgoNew::unwrapSuperCluster()")
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
    throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
      << "No associated block elements to SuperCluster! Shouldn't happen!"
      << std::endl;
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
}



int PFEGammaAlgoNew::attachPSClusters(const PFSCElement* thesc,
				      const ClusterElement* ecalclus,
				      ClusterMap::mapped_type& eslist) {  
  if( ecalclus->clusterRef()->layer() == PFLayer::ECAL_BARREL ) return 0;
  SuperClusterRef::key_type sc_key = thesc->superClusterRef().key();
  edm::Ptr<reco::CaloCluster> clusptr = refToPtr(ecalclus->clusterRef());
  EEtoPSElement ecalkey = std::make_pair(clusptr.key(),clusptr);
  const EEtoPSAssociation::value_type& psmap = eetops_->at(sc_key);
  auto assc_ps = std::equal_range(psmap.cbegin(),
				  psmap.cend(),
				  ecalkey,comparePSMapByKey);
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

int PFEGammaAlgoNew::attachPSClusters(const ClusterElement* ecalclus,
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
    throw cms::Exception("PFEGammaAlgoNew::attachPSClusters()")
      << "Didn't insert correct number of ps clusters!";
  }
  return eslist.size();
}

void PFEGammaAlgoNew::dumpCurrentRefinableObjects() const {
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
void PFEGammaAlgoNew::
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
      std::sort(ecalbegin,ecalend,closestTrackToECAL);
      PFFlaggedElement& closestECAL = 
	_splayedblock[reco::PFBlockElement::ECAL].front();
      const double dist = _currentblock->dist(kftrack.first->index(), 
					      closestECAL.first->index(),
					      _currentlinks,
					      reco::PFBlock::LINKTEST_ALL);
      if( dist != -1.0 && closestECAL.second ) {
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
	if( !gsflinked ) { // determine if we should remove the matched cluster
	  const reco::PFBlockElementTrack * kfEle = 
	    docast(const reco::PFBlockElementTrack*,kftrack.first);
	  const reco::TrackRef trackref = kfEle->trackRef();
	  const unsigned Algo = trackref->algo();
	  const int nexhits = 
	    trackref->trackerExpectedHitsInner().numberOfLostHits();
	  bool fromprimaryvertex = false;
	  for( auto vtxtks = primaryVertex_->tracks_begin();
	       vtxtks != primaryVertex_->tracks_end(); ++ vtxtks ) {
	    if( trackref == vtxtks->castTo<reco::TrackRef>() ) {
	      fromprimaryvertex = true;
	      break;
	    }
	  }// loop over tracks in primary vertex
	   // if associated to good non-GSF matched track remove this cluster
	  if( Algo < 9 && nexhits == 0 && fromprimaryvertex ) {
	    closestECAL.second = false;
	  } else { // otherwise associate the cluster and KF track
	    _recoveredlinks.emplace(closestECAL.first,kftrack.first);
	    _recoveredlinks.emplace(kftrack.first,closestECAL.first);
	  }
	}
      } // found a good closest ECAL match
    } // no GSF track matched to KF
  } // loop over KF elements
}

void PFEGammaAlgoNew::
mergeROsByAnyLink(std::list<PFEGammaAlgoNew::ProtoEGObject>& ROs) {
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
void PFEGammaAlgoNew::
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
	RO.localMap.emplace(seedtk,elemaskf);
	RO.localMap.emplace(elemaskf,seedtk);
      } else if ( elemaskf->trackType(convType) ) {
	kft->second = false;
	RO.secondaryKFs.push_back(std::make_pair(elemaskf,true));
	RO.localMap.emplace(seedtk,elemaskf);
	RO.localMap.emplace(elemaskf,seedtk);
      }
    }// loop on closest KFs not closer to other GSFs
  } // loop on GSF primaries on RO  
}

void PFEGammaAlgoNew::
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
      throw cms::Exception("PFEGammaAlgoNew::linkRefinableObjectPrimaryKFsToSecondaryKFs()")
	<< "A KF track from conversion has been assigned as a primary!!"
	<< std::endl;
    }
    NotCloserToOther<reco::PFBlockElement::TRACK,reco::PFBlockElement::TRACK>
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
	RO.localMap.emplace(primkf,elemaskf);
	RO.localMap.emplace(elemaskf,primkf);
      } 
    }// loop on closest KFs not closer to other KFs
  } // loop on KF primaries on RO
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgoNew::
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
      RO.ecalclusters.push_front(temp);
      RO.localMap.emplace(primgsf.first,temp.first);
      RO.localMap.emplace(temp.first,primgsf.first);
      ECALbegin->second = false;
    } else {
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") 
	<< "Found a cluster already associated to GSF extrapolation"
	<< " at ECAL surface!" << std::endl;
      RO.localMap.emplace(primgsf.first,temp.first);
      RO.localMap.emplace(temp.first,primgsf.first);
      std::swap(*RO.ecalclusters.begin(),*cluster);
    }
  }
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgoNew::
linkRefinableObjectKFTracksToECAL(ProtoEGObject& RO) {
  if( !_splayedblock[reco::PFBlockElement::ECAL].size() ) return;  
  for( auto& primkf : RO.primaryKFs ) linkKFTrackToECAL(primkf,RO);
  for( auto& secdkf : RO.secondaryKFs ) linkKFTrackToECAL(secdkf,RO);
}

void PFEGammaAlgoNew::linkKFTrackToECAL(const KFFlaggedElement& kfflagged,
					ProtoEGObject& RO) {
  std::list<PFClusterFlaggedElement>& currentECAL = RO.ecalclusters;
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
    RO.localMap.emplace(elemascluster,kfflagged.first);
    RO.localMap.emplace(kfflagged.first,elemascluster);
  }
}

void PFEGammaAlgoNew::
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
	RO.localMap.emplace(ecal->first,bremflagged.first);
	RO.localMap.emplace(bremflagged.first,ecal->first);
	LOGDRESSED("PFEGammaAlgo::linkBremToECAL()") 
	  << "Found a cluster not already associated by brem extrapolation"
	  << " at ECAL surface!" << std::endl;
      }
    }
  }  
}

void PFEGammaAlgoNew::
linkRefinableObjectConvSecondaryKFsToSecondaryKFs(ProtoEGObject& RO) {
  IsConversionTrack<reco::PFBlockElementTrack> isConvKf; 
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  auto BeginROskfs = RO.secondaryKFs.begin();
  auto EndROskfs   = RO.secondaryKFs.end();
  auto ronotconv = std::partition(BeginROskfs,EndROskfs,isConvKf); 
  for( auto& ro_skf = BeginROskfs; ro_skf != ronotconv; ++ro_skf ) {
    NotCloserToOther<reco::PFBlockElement::TRACK,
                     reco::PFBlockElement::TRACK,
                     true>
      TracksToTracks(_currentblock,_currentlinks,ro_skf->first); 
    auto notmatched = std::partition(KFbegin,KFend,TracksToTracks);
    notmatched = std::partition(KFbegin,notmatched,isConvKf);
    for( auto kf = KFbegin; kf != notmatched; ++kf ) {
      const reco::PFBlockElementTrack* elemaskf =
	docast(const reco::PFBlockElementTrack*,kf->first);
      RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
      RO.localMap.emplace(ro_skf->first,elemaskf);
      RO.localMap.emplace(elemaskf,ro_skf->first);
      kf->second = false;
    }
  }
}

void PFEGammaAlgoNew::
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
      RO.localMap.emplace(ecal.first,elemaskf);
      RO.localMap.emplace(elemaskf,ecal.first);
      kf->second = false;
    }
    // go through non-conv-identified kfs and check MVA to add conversions
    for( auto kf = notconvkf; kf != notmatchedkf; ++kf ) {
      if( EvaluateSingleLegMVA(_currentblock, *primaryVertex_, 
			       kf->first->index()) ) {
	const reco::PFBlockElementTrack* elemaskf =
	  docast(const reco::PFBlockElementTrack*,kf->first);
	RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
	RO.localMap.emplace(ecal.first,elemaskf);
	RO.localMap.emplace(elemaskf,ecal.first);
	kf->second = false;
      }
    }    
  }
}

void  PFEGammaAlgoNew::
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
      RO.ecalclusters.push_back( std::make_pair(elemascluster,true) );
      RO.localMap.emplace(skf.first,elemascluster);
      RO.localMap.emplace(elemascluster,skf.first);
      ecal->second = false;
    }
  }
}

void PFEGammaAlgoNew::
fillPFCandidates(std::list<PFEGammaAlgoNew::ProtoEGObject>& ROs) {
  
}

unsigned int PFEGammaAlgoNew::whichTrackAlgo(const reco::TrackRef& trackRef) {
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
bool PFEGammaAlgoNew::isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
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

void PFEGammaAlgoNew::AddElectronElements(unsigned int gsf_index,
			             std::vector<unsigned int> &elemsToLock,
				     const reco::PFBlockRef&  blockRef,
				     AsscMap& associatedToGsf_,
				     AsscMap& associatedToBrems_,
				     AsscMap& associatedToEcal_){
  const reco::PFBlock& block = *blockRef;
  PFBlock::LinkData linkData =  block.linkData();  
   
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  
  const reco::PFBlockElementGsfTrack * GsfEl  =  
    docast(const reco::PFBlockElementGsfTrack*,(&elements[gsf_index]));
  reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();

  // lock only the elements that pass the BDT cut
//   bool bypassmva=false;
//   if(useEGElectrons_) {
//     GsfElectronEqual myEqual(RefGSF);
//     std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(theGsfElectrons_->begin(),theGsfElectrons_->end(),myEqual);
//     if(itcheck!=theGsfElectrons_->end()) {
//       if(BDToutput_[cgsf] >= -1.) 
// 	bypassmva=true;
//     }
//   }

  //if(BDToutput_[cgsf] < mvaEleCut_ && bypassmva == false) continue;

  
  elemsToLock.push_back(gsf_index);
  vector<unsigned int> &assogsf_index = associatedToGsf_[gsf_index];
  for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
    PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
    // lock the elements associated to the gsf: ECAL, Brems
    elemsToLock.push_back((assogsf_index[ielegsf]));
    if (assoele_type == reco::PFBlockElement::ECAL) {
      unsigned int keyecalgsf = assogsf_index[ielegsf];

      // added protection against fifth step
      if(fifthStepKfTrack_.size() > 0) {
	for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
	  if(fifthStepKfTrack_[itr].first == keyecalgsf) {
	    elemsToLock.push_back((fifthStepKfTrack_[itr].second));
	  }
	}
      }

      // added locking for conv gsf tracks and kf tracks
      if(convGsfTrack_.size() > 0) {
	for(unsigned int iconv = 0; iconv < convGsfTrack_.size(); iconv++) {
	  if(convGsfTrack_[iconv].first == keyecalgsf) {
	    // lock the GSF track
	    elemsToLock.push_back(convGsfTrack_[iconv].second);
	    // lock also the KF track associated
	    std::multimap<double, unsigned> convKf;
	    block.associatedElements( convGsfTrack_[iconv].second,
				      linkData,
				      convKf,
				      reco::PFBlockElement::TRACK,
				      reco::PFBlock::LINKTEST_ALL );
	    if(convKf.size() > 0) {
	      elemsToLock.push_back(convKf.begin()->second);
	    }
	  }
	}
      }


      vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(keyecalgsf)->second;
      for(unsigned int ips =0; ips<assoecalgsf_index.size();ips++) {
	// lock the elements associated to ECAL: PS1,PS2, for the moment not HCAL
	if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::PS1) 
	  elemsToLock.push_back((assoecalgsf_index[ips]));
	if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::PS2) 
	  elemsToLock.push_back(assoecalgsf_index[ips]);
	if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::TRACK) {
	  //FIXME: some extra input needed here which is not available yet
// 	  if(lockExtraKf_[cgsf] == true) {	      
// 	    elemsToLock.push_back(assoecalgsf_index[ips])
// 	  }
	}
      }
    } // End if ECAL
    if (assoele_type == reco::PFBlockElement::BREM) {
      unsigned int brem_index = assogsf_index[ielegsf];
      vector<unsigned int> assobrem_index = associatedToBrems_.find(brem_index)->second;
      for (unsigned int ibrem = 0; ibrem < assobrem_index.size(); ibrem++){
	if (elements[(assobrem_index[ibrem])].type() == reco::PFBlockElement::ECAL) {
	  unsigned int keyecalbrem = assobrem_index[ibrem];
	  // lock the ecal cluster associated to the brem
	  elemsToLock.push_back(assobrem_index[ibrem]);

	  // add protection against fifth step
	  if(fifthStepKfTrack_.size() > 0) {
	    for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
	      if(fifthStepKfTrack_[itr].first == keyecalbrem) {
		elemsToLock.push_back(fifthStepKfTrack_[itr].second);
	      }
	    }
	  }

	  vector<unsigned int> assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	  // lock the elements associated to ECAL: PS1,PS2, for the moment not HCAL
	  for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
	    if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) 
	      elemsToLock.push_back(assoelebrem_index[ielebrem]);
	    if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) 
	      elemsToLock.push_back(assoelebrem_index[ielebrem]);
	  }
	}
      }
    } // End if BREM	  
  } // End loop on elements from gsf track
  return;
}

// This function get the associatedToGsf and associatedToBrems maps and  
// compute the electron 4-mom and set the pf candidate, for
// the gsf track with a BDTcut > mvaEleCut_
bool PFEGammaAlgoNew::AddElectronCandidate(unsigned int gsf_index,
					reco::SuperClusterRef scref,
					 std::vector<unsigned int> &elemsToLock,
					 const reco::PFBlockRef&  blockRef,
					 AsscMap& associatedToGsf_,
					 AsscMap& associatedToBrems_,
					 AsscMap& associatedToEcal_,
					 std::vector<bool>& active) {
  
  const reco::PFBlock& block = *blockRef;
  PFBlock::LinkData linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  PFEnergyResolution pfresol_;
  //PFEnergyCalibration pfcalib_;

  bool DebugIDCandidates = false;
//   vector<reco::PFCluster> pfClust_vec(0);
//   pfClust_vec.clear();

	  
  // They should be reset for each gsf track
  int eecal=0;
  int hcal=0;
  int charge =0; 
  // bool goodphi=true;
  math::XYZTLorentzVector momentum_kf,momentum_gsf,momentum,momentum_mean;
  float dpt=0; float dpt_gsf=0;
  float Eene=0; float dene=0; float Hene=0.;
  float RawEene = 0.;
  double posX=0.;
  double posY=0.;
  double posZ=0.;
  std::vector<float> bremEnergyVec;

  std::vector<const PFCluster*> pfSC_Clust_vec; 

  float de_gs = 0., de_me = 0., de_kf = 0.; 
  float m_el=0.00051;
  int nhit_kf=0; int nhit_gsf=0;
  bool has_gsf=false;
  bool has_kf=false;
  math::XYZTLorentzVector newmomentum;
  float ps1TotEne = 0;
  float ps2TotEne = 0;
  vector<unsigned int> elementsToAdd(0);
  reco::TrackRef RefKF;  



  elementsToAdd.push_back(gsf_index);
  const reco::PFBlockElementGsfTrack * GsfEl  =  
    docast(const reco::PFBlockElementGsfTrack*,(&elements[gsf_index]));
  const math::XYZPointF& posGsfEcalEntrance = GsfEl->positionAtECALEntrance();
  reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();
  if (RefGSF.isNonnull()) {
    
    has_gsf=true;
    
    charge= RefGSF->chargeMode();
    nhit_gsf= RefGSF->hitPattern().trackerLayersWithMeasurement();
    
    momentum_gsf.SetPx(RefGSF->pxMode());
    momentum_gsf.SetPy(RefGSF->pyMode());
    momentum_gsf.SetPz(RefGSF->pzMode());
    float ENE=sqrt(RefGSF->pMode()*
		    RefGSF->pMode()+m_el*m_el);
    
    if( DebugIDCandidates ) 
      cout << "SetCandidates:: GsfTrackRef: Ene " << ENE 
	    << " charge " << charge << " nhits " << nhit_gsf <<endl;
    
    momentum_gsf.SetE(ENE);       
    dpt_gsf=RefGSF->ptModeError()*
      (RefGSF->pMode()/RefGSF->ptMode());
    
    momentum_mean.SetPx(RefGSF->px());
    momentum_mean.SetPy(RefGSF->py());
    momentum_mean.SetPz(RefGSF->pz());
    float ENEm=sqrt(RefGSF->p()*
		    RefGSF->p()+m_el*m_el);
    momentum_mean.SetE(ENEm);       
    //       dpt_mean=RefGSF->ptError()*
    // 	(RefGSF->p()/RefGSF->pt());  
  }
  else {
    if( DebugIDCandidates ) 
      cout <<  "SetCandidates:: !!!!  NULL GSF Track Ref " << endl;	
  } 

  //    vector<unsigned int> assogsf_index =  associatedToGsf_[igsf].second;
  vector<unsigned int> &assogsf_index = associatedToGsf_[gsf_index];
  unsigned int ecalGsf_index = 100000;
  bool FirstEcalGsf = true;
  for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
    PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
    if  (assoele_type == reco::PFBlockElement::TRACK) {
      elementsToAdd.push_back((assogsf_index[ielegsf])); // Daniele
      const reco::PFBlockElementTrack * KfTk =  
	docast(const reco::PFBlockElementTrack*,
	       (&elements[(assogsf_index[ielegsf])]));
      // 19 Mar 2010 do not consider here track from gamam conv
      bool isPrim = isPrimaryTrack(*KfTk,*GsfEl);
      if(!isPrim) continue;
      
      RefKF = KfTk->trackRef();
      if (RefKF.isNonnull()) {
	has_kf = true;
	// dpt_kf=(RefKF->ptError()*RefKF->ptError());
	nhit_kf=RefKF->hitPattern().trackerLayersWithMeasurement();
	momentum_kf.SetPx(RefKF->px());
	momentum_kf.SetPy(RefKF->py());
	momentum_kf.SetPz(RefKF->pz());
	float ENE=sqrt(RefKF->p()*RefKF->p()+m_el*m_el);
	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: KFTrackRef: Ene " << ENE << " nhits " << nhit_kf << endl;
	
	momentum_kf.SetE(ENE);
      }
      else {
	if( DebugIDCandidates ) 
	  cout <<  "SetCandidates:: !!!! NULL KF Track Ref " << endl;
      }
    } 

    if  (assoele_type == reco::PFBlockElement::ECAL) {
      unsigned int keyecalgsf = assogsf_index[ielegsf];
      vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(keyecalgsf)->second;
      vector<double> ps1Ene(0);
      vector<double> ps2Ene(0);
      // Important is the PS clusters are not saved before the ecal one, these
      // energy are not correctly assigned 
      // For the moment I get only the closest PS clusters: this has to be changed
      for(unsigned int ips =0; ips<assoecalgsf_index.size();ips++) {
	PFBlockElement::Type typeassoecal = elements[(assoecalgsf_index[ips])].type();
	if  (typeassoecal == reco::PFBlockElement::PS1) {  
	  PFClusterRef  psref = elements[(assoecalgsf_index[ips])].clusterRef();
	  ps1Ene.push_back(psref->energy());
	  elementsToAdd.push_back((assoecalgsf_index[ips]));
	}
	if  (typeassoecal == reco::PFBlockElement::PS2) {  
	  PFClusterRef  psref = elements[(assoecalgsf_index[ips])].clusterRef();
	  ps2Ene.push_back(psref->energy());
	  elementsToAdd.push_back((assoecalgsf_index[ips]));
	}
	if  (typeassoecal == reco::PFBlockElement::HCAL) {
	  const reco::PFBlockElementCluster * clust =  
	    docast(const reco::PFBlockElementCluster*,
		   (&elements[(assoecalgsf_index[ips])])); 
	  elementsToAdd.push_back((assoecalgsf_index[ips])); 
	  Hene+=clust->clusterRef()->energy();
	  hcal++;
	}
      }
      elementsToAdd.push_back((assogsf_index[ielegsf]));


      const reco::PFBlockElementCluster * clust =  
	docast(const reco::PFBlockElementCluster*,
	       (&elements[(assogsf_index[ielegsf])]));
      
      eecal++;
      
      const reco::PFCluster& cl(*clust->clusterRef());
      //pfClust_vec.push_back((*clust->clusterRef()));

      // The electron RAW energy is the energy of the corrected GSF cluster	
      double ps1,ps2;
      ps1=ps2=0.;
      //	float EE=pfcalib_.energyEm(cl,ps1Ene,ps2Ene);
      float EE = thePFEnergyCalibration_->energyEm(cl,ps1Ene,ps2Ene,ps1,ps2,applyCrackCorrections_);	  
      //	float RawEE = cl.energy();

      float ceta=cl.position().eta();
      float cphi=cl.position().phi();
      
      /*
	float mphi=-2.97025;
	if (ceta<0) mphi+=0.00638;
	
	for (int ip=1; ip<19; ip++){
	float df= cphi - (mphi+(ip*6.283185/18));
	if (fabs(df)<0.01) goodphi=false;
	}
      */

      float dE=pfresol_.getEnergyResolutionEm(EE,cl.position().eta());
      if( DebugIDCandidates ) 
	cout << "SetCandidates:: EcalCluster: EneNoCalib " << clust->clusterRef()->energy()  
	      << " eta,phi " << ceta << "," << cphi << " Calib " <<  EE << " dE " <<  dE <<endl;

      bool elecCluster=false;
      if (FirstEcalGsf) {
	FirstEcalGsf = false;
	elecCluster=true;
	ecalGsf_index = assogsf_index[ielegsf];
	//	  std::cout << " PFElectronAlgo / Seed " << EE << std::endl;
	RawEene += EE;
      }
      
      // create a photon/electron candidate
      math::XYZTLorentzVector clusterMomentum;
      math::XYZPoint direction=cl.position()/cl.position().R();
      clusterMomentum.SetPxPyPzE(EE*direction.x(),
				  EE*direction.y(),
				  EE*direction.z(),
				  EE);
      reco::PFCandidate cluster_Candidate((elecCluster)?charge:0,
					  clusterMomentum, 
					  (elecCluster)? reco::PFCandidate::e : reco::PFCandidate::gamma);
      
      cluster_Candidate.setPs1Energy(ps1);
      cluster_Candidate.setPs2Energy(ps2);
      // The Raw Ecal energy will be the energy of the basic cluster. 
      // It will be the corrected energy without the preshower
      cluster_Candidate.setEcalEnergy(EE-ps1-ps2,EE);
      //	      std::cout << " PFElectronAlgo, adding Brem (1) " << EE << std::endl;
      cluster_Candidate.setPositionAtECALEntrance(math::XYZPointF(cl.position()));
      cluster_Candidate.addElementInBlock(blockRef,assogsf_index[ielegsf]);
      // store the photon candidate
//       std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
// 	electronConstituents_.find(cgsf);
//       if(itcheck==electronConstituents_.end())
// 	{		  
// 	  // beurk
// 	  std::vector<reco::PFCandidate> tmpVec;
// 	  tmpVec.push_back(cluster_Candidate);
// 	  electronConstituents_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
// 					(cgsf,tmpVec));
// 	}
//       else
// 	{
// 	  itcheck->second.push_back(cluster_Candidate);
// 	}
      
      Eene+=EE;
      posX +=  EE * cl.position().X();
      posY +=  EE * cl.position().Y();
      posZ +=  EE * cl.position().Z();	  
      ps1TotEne+=ps1;
      ps2TotEne+=ps2;
      dene+=dE*dE;
      
      //MM Add cluster to the vector pfSC_Clust_vec needed for brem corrections
      pfSC_Clust_vec.push_back( &cl );

    }
    


    // Important: Add energy from the brems
    if  (assoele_type == reco::PFBlockElement::BREM) {
      unsigned int brem_index = assogsf_index[ielegsf];
      vector<unsigned int> assobrem_index = associatedToBrems_.find(brem_index)->second;
      elementsToAdd.push_back(brem_index);
      for (unsigned int ibrem = 0; ibrem < assobrem_index.size(); ibrem++){
	if (elements[(assobrem_index[ibrem])].type() == reco::PFBlockElement::ECAL) {
	  // brem emission is from the considered gsf track
	  if( assobrem_index[ibrem] !=  ecalGsf_index) {
	    unsigned int keyecalbrem = assobrem_index[ibrem];
	    const vector<unsigned int>& assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	    vector<double> ps1EneFromBrem(0);
	    vector<double> ps2EneFromBrem(0);
	    for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) {
		PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		ps1EneFromBrem.push_back(psref->energy());
		elementsToAdd.push_back(assoelebrem_index[ielebrem]);
	      }
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) {
		PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		ps2EneFromBrem.push_back(psref->energy());
		elementsToAdd.push_back(assoelebrem_index[ielebrem]);
	      }	  
	    }
	    elementsToAdd.push_back(assobrem_index[ibrem]);
	    reco::PFClusterRef clusterRef = elements[(assobrem_index[ibrem])].clusterRef();
	    //pfClust_vec.push_back(*clusterRef);
	    // to get a calibrated PS energy 
	    double ps1=0;
	    double ps2=0;
	    float EE = thePFEnergyCalibration_->energyEm(*clusterRef,ps1EneFromBrem,ps2EneFromBrem,ps1,ps2,applyCrackCorrections_);
	    bremEnergyVec.push_back(EE);
	    // float RawEE  = clusterRef->energy();
	    float ceta = clusterRef->position().eta();
	    // float cphi = clusterRef->position().phi();
	    float dE=pfresol_.getEnergyResolutionEm(EE,ceta);
	    if( DebugIDCandidates ) 
	      cout << "SetCandidates:: BremCluster: Ene " << EE << " dE " <<  dE <<endl;	  

	    Eene+=EE;
	    posX +=  EE * clusterRef->position().X();
	    posY +=  EE * clusterRef->position().Y();
	    posZ +=  EE * clusterRef->position().Z();	  
	    ps1TotEne+=ps1;
	    ps2TotEne+=ps2;
	    // Removed 4 March 2009. Florian. The Raw energy is the (corrected) one of the GSF cluster only
	    //	      RawEene += RawEE;
	    dene+=dE*dE;

	    //MM Add cluster to the vector pfSC_Clust_vec needed for brem corrections
	    pfSC_Clust_vec.push_back( clusterRef.get() );

	    // create a PFCandidate out of it. Watch out, it is for the e/gamma and tau only
	    // not to be used by the PFAlgo
	    math::XYZTLorentzVector photonMomentum;
	    math::XYZPoint direction=clusterRef->position()/clusterRef->position().R();
	    
	    photonMomentum.SetPxPyPzE(EE*direction.x(),
				      EE*direction.y(),
				      EE*direction.z(),
				      EE);
	    reco::PFCandidate photon_Candidate(0,photonMomentum, reco::PFCandidate::gamma);
	    
	    photon_Candidate.setPs1Energy(ps1);
	    photon_Candidate.setPs2Energy(ps2);
	    // yes, EE, we want the raw ecal energy of the daugther to have the same definition
	    // as the GSF cluster
	    photon_Candidate.setEcalEnergy(EE-ps1-ps2,EE);
	    //	      std::cout << " PFElectronAlgo, adding Brem " << EE << std::endl;
	    photon_Candidate.setPositionAtECALEntrance(math::XYZPointF(clusterRef->position()));
	    photon_Candidate.addElementInBlock(blockRef,assobrem_index[ibrem]);

	    // store the photon candidate
	    //FIXME: constituents needed?
// 	    std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
// 	      electronConstituents_.find(cgsf);
// 	    if(itcheck==electronConstituents_.end())
// 	      {		  
// 		// beurk
// 		std::vector<reco::PFCandidate> tmpVec;
// 		tmpVec.push_back(photon_Candidate);
// 		electronConstituents_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
// 					  (cgsf,tmpVec));
// 	      }
// 	    else
// 	      {
// 		itcheck->second.push_back(photon_Candidate);
// 	      }
	  }
	} 
      }
    }
  } // End Loop On element associated to the GSF tracks
  if (has_gsf) {
    
    // SuperCluster energy corrections
    double unCorrEene = Eene;
    double absEta = fabs(momentum_gsf.Eta());
    double emTheta = momentum_gsf.Theta();
    PFClusterWidthAlgo pfSCwidth(pfSC_Clust_vec); 
    double brLinear = pfSCwidth.pflowPhiWidth()/pfSCwidth.pflowEtaWidth(); 
    pfSC_Clust_vec.clear();
    
    if( DebugIDCandidates ) 
      cout << "PFEelectronAlgo:: absEta " << absEta  << " theta " << emTheta 
	    << " EneRaw " << Eene << " Err " << dene;
    
    // The calibrations are provided till ET = 200 GeV //No longer a such cut MM
    // Protection on at least 1 GeV energy...avoid possible divergencies at very low energy.
    if(usePFSCEleCalib_ && unCorrEene > 0.) { 
      if( absEta < 1.5) {
	double Etene = Eene*sin(emTheta);
	double emBR_e = thePFSCEnergyCalibration_->SCCorrFBremBarrel(Eene, Etene, brLinear); 
	double emBR_et = emBR_e*sin(emTheta); 
	double emCorrFull_et = thePFSCEnergyCalibration_->SCCorrEtEtaBarrel(emBR_et, absEta); 
	Eene = emCorrFull_et/sin(emTheta);
      }
      else {
	//  double Etene = Eene*sin(emTheta); //not needed anymore for endcaps MM
	double emBR_e = thePFSCEnergyCalibration_->SCCorrFBremEndcap(Eene, absEta, brLinear); 
	double emBR_et = emBR_e*sin(emTheta); 
	double emCorrFull_et = thePFSCEnergyCalibration_->SCCorrEtEtaEndcap(emBR_et, absEta); 
	Eene = emCorrFull_et/sin(emTheta);
      }
      dene = sqrt(dene)*(Eene/unCorrEene);
      dene = dene*dene;
    }

    if( DebugIDCandidates ) 
      cout << " EneCorrected " << Eene << " Err " << dene  << endl;

    // charge determination with the majority method
    // if the kf track exists: 2 among 3 of supercluster barycenter position
    // gsf track and kf track
    if(has_kf && unCorrEene > 0.) {
      posX /=unCorrEene;
      posY /=unCorrEene;
      posZ /=unCorrEene;
      math::XYZPoint sc_pflow(posX,posY,posZ);

      std::multimap<double, unsigned int> bremElems;
      block.associatedElements( gsf_index,linkData,
				bremElems,
				reco::PFBlockElement::BREM,
				reco::PFBlock::LINKTEST_ALL );

      double phiTrack = RefGSF->phiMode();
      if(bremElems.size()>0) {
	unsigned int brem_index =  bremElems.begin()->second;
	const reco::PFBlockElementBrem * BremEl  =  
	  docast(const reco::PFBlockElementBrem*,(&elements[brem_index]));
	phiTrack = BremEl->positionAtECALEntrance().phi();
      }

      double dphi_normalsc = sc_pflow.Phi() - phiTrack;
      if ( dphi_normalsc < -M_PI ) 
	dphi_normalsc = dphi_normalsc + 2.*M_PI;
      else if ( dphi_normalsc > M_PI ) 
	dphi_normalsc = dphi_normalsc - 2.*M_PI;
      
      int chargeGsf = RefGSF->chargeMode();
      int chargeKf = RefKF->charge();

      int chargeSC = 0;
      if(dphi_normalsc < 0.) 
	chargeSC = 1;
      else 
	chargeSC = -1;
      
      if(chargeKf == chargeGsf) 
	charge = chargeGsf;
      else if(chargeGsf == chargeSC)
	charge = chargeGsf;
      else 
	charge = chargeKf;

      if( DebugIDCandidates ) 
	cout << "PFElectronAlgo:: charge determination " 
	      << " charge GSF " << chargeGsf 
	      << " charge KF " << chargeKf 
	      << " charge SC " << chargeSC
	      << " Final Charge " << charge << endl;
      
    }
      
    // Think about this... 
    if ((nhit_gsf<8) && (has_kf)){
      
      // Use Hene if some condition.... 
      
      momentum=momentum_kf;
      float Fe=Eene;
      float scale= Fe/momentum.E();
      
      // Daniele Changed
      if (Eene < 0.0001) {
	Fe = momentum.E();
	scale = 1.;
      }


      newmomentum.SetPxPyPzE(scale*momentum.Px(),
			      scale*momentum.Py(),
			      scale*momentum.Pz(),Fe);
      if( DebugIDCandidates ) 
	cout << "SetCandidates:: (nhit_gsf<8) && (has_kf):: pt " << newmomentum.pt() << " Ene " <<  Fe <<endl;

      
    } 
    if ((nhit_gsf>7) || (has_kf==false)){
      if(Eene > 0.0001) {
	de_gs=1-momentum_gsf.E()/Eene;
	de_me=1-momentum_mean.E()/Eene;
	de_kf=1-momentum_kf.E()/Eene;
      }

      momentum=momentum_gsf;
      dpt=1/(dpt_gsf*dpt_gsf);
      
      if(dene > 0.)
	dene= 1./dene;
      
      float Fe = 0.;
      if(Eene > 0.0001) {
	Fe =((dene*Eene) +(dpt*momentum.E()))/(dene+dpt);
      }
      else {
	Fe=momentum.E();
      }
      
      if ((de_gs>0.05)&&(de_kf>0.05)){
	Fe=Eene;
      }
      if ((de_gs<-0.1)&&(de_me<-0.1) &&(de_kf<0.) && 
	  (momentum.E()/dpt_gsf) > 5. && momentum_gsf.pt() < 30.){
	Fe=momentum.E();
      }
      float scale= Fe/momentum.E();
      
      newmomentum.SetPxPyPzE(scale*momentum.Px(),
			      scale*momentum.Py(),
			      scale*momentum.Pz(),Fe);
      if( DebugIDCandidates ) 
	cout << "SetCandidates::(nhit_gsf>7) || (has_kf==false)  " << newmomentum.pt() << " Ene " <<  Fe <<endl;
      
      
    }
    if (newmomentum.pt()>0.5){
      
      // the pf candidate are created: we need to set something more? 
      // IMPORTANT -> We need the gsftrackRef, not only the TrackRef??

      if( DebugIDCandidates )
	cout << "SetCandidates:: I am before doing candidate " <<endl;
      
      //vector with the cluster energies (for the extra)
      std::vector<float> clusterEnergyVec;
      clusterEnergyVec.push_back(RawEene);
      clusterEnergyVec.insert(clusterEnergyVec.end(),bremEnergyVec.begin(),bremEnergyVec.end());

      // add the information in the extra
      //std::vector<reco::PFCandidateElectronExtra>::iterator itextra;
      //PFElectronExtraEqual myExtraEqual(RefGSF);
      PFCandidateEGammaExtra myExtraEqual(RefGSF);
      //myExtraEqual.setSuperClusterRef(scref);
      myExtraEqual.setSuperClusterBoxRef(scref);
      myExtraEqual.setClusterEnergies(clusterEnergyVec);
      //itextra=find_if(electronExtra_.begin(),electronExtra_.end(),myExtraEqual);
      //if(itextra!=electronExtra_.end()) {
	//itextra->setClusterEnergies(clusterEnergyVec);
//       else {
// 	if(RawEene>0.) 
// 	  std::cout << " There is a big problem with the electron extra, PFElectronAlgo should crash soon " << RawEene << std::endl;
//       }

      reco::PFCandidate::ParticleType particleType 
	= reco::PFCandidate::e;
      //reco::PFCandidate temp_Candidate;
      reco::PFCandidate temp_Candidate(charge,newmomentum,particleType);
      //FIXME: need bdt output
      //temp_Candidate.set_mva_e_pi(BDToutput_[cgsf]);
      temp_Candidate.setEcalEnergy(RawEene,Eene);
      // Note the Hcal energy is set but the element is never locked 
      temp_Candidate.setHcalEnergy(Hene,Hene);  
      temp_Candidate.setPs1Energy(ps1TotEne);
      temp_Candidate.setPs2Energy(ps2TotEne);
      temp_Candidate.setTrackRef(RefKF);   
      // This reference could be NULL it is needed a protection? 
      temp_Candidate.setGsfTrackRef(RefGSF);
      temp_Candidate.setPositionAtECALEntrance(posGsfEcalEntrance);
      // Add Vertex
      temp_Candidate.setVertexSource(PFCandidate::kGSFVertex);
      
      //supercluster ref is always available now and points to ecal-drive box/mustache supercluster
      temp_Candidate.setSuperClusterRef(scref);
      
      // save the superclusterRef when available
      //FIXME: Point back to ecal-driven supercluster ref, which is now always available
//       if(RefGSF->extra().isAvailable() && RefGSF->extra()->seedRef().isAvailable()) {
// 	reco::ElectronSeedRef seedRef=  RefGSF->extra()->seedRef().castTo<reco::ElectronSeedRef>();
// 	if(seedRef.isAvailable() && seedRef->isEcalDriven()) {
// 	  reco::SuperClusterRef scRef = seedRef->caloCluster().castTo<reco::SuperClusterRef>();
// 	  if(scRef.isNonnull())  
// 	    temp_Candidate.setSuperClusterRef(scRef);
// 	}
//       }

      if( DebugIDCandidates ) 
	cout << "SetCandidates:: I am after doing candidate " <<endl;
      
//       for (unsigned int elad=0; elad<elementsToAdd.size();elad++){
// 	temp_Candidate.addElementInBlock(blockRef,elementsToAdd[elad]);
//       }
// 
//       // now add the photons to this candidate
//       std::map<unsigned int, std::vector<reco::PFCandidate> >::const_iterator itcluster=
// 	electronConstituents_.find(cgsf);
//       if(itcluster!=electronConstituents_.end())
// 	{
// 	  const std::vector<reco::PFCandidate> & theClusters=itcluster->second;
// 	  unsigned nclus=theClusters.size();
// 	  //	    std::cout << " PFElectronAlgo " << nclus << " daugthers to add" << std::endl;
// 	  for(unsigned iclus=0;iclus<nclus;++iclus)
// 	    {
// 	      temp_Candidate.addDaughter(theClusters[iclus]);
// 	    }
// 	}

      // By-pass the mva is the electron has been pre-selected 
//       bool bypassmva=false;
//       if(useEGElectrons_) {
// 	GsfElectronEqual myEqual(RefGSF);
// 	std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(theGsfElectrons_->begin(),theGsfElectrons_->end(),myEqual);
// 	if(itcheck!=theGsfElectrons_->end()) {
// 	  if(BDToutput_[cgsf] >= -1.)  {
// 	    // bypass the mva only if the reconstruction went fine
// 	    bypassmva=true;
// 
// 	    if( DebugIDCandidates ) {
// 	      if(BDToutput_[cgsf] < -0.1) {
// 		float esceg = itcheck->caloEnergy();		
// 		cout << " Attention By pass the mva " << BDToutput_[cgsf] 
// 		      << " SuperClusterEnergy " << esceg
// 		      << " PF Energy " << Eene << endl;
// 		
// 		cout << " hoe " << itcheck->hcalOverEcal()
// 		      << " tkiso04 " << itcheck->dr04TkSumPt()
// 		      << " ecaliso04 " << itcheck->dr04EcalRecHitSumEt()
// 		      << " hcaliso04 " << itcheck->dr04HcalTowerSumEt()
// 		      << " tkiso03 " << itcheck->dr03TkSumPt()
// 		      << " ecaliso03 " << itcheck->dr03EcalRecHitSumEt()
// 		      << " hcaliso03 " << itcheck->dr03HcalTowerSumEt() << endl;
// 	      }
// 	    } // end DebugIDCandidates
// 	  }
// 	}
//       }
      
      myExtraEqual.setStatus(PFCandidateEGammaExtra::Selected,true);
      
      // ... and lock all elemts used
      for(std::vector<unsigned int>::const_iterator it = elemsToLock.begin();
	  it != elemsToLock.end(); ++it)
	{
	  if(active[*it])
	    {
	      temp_Candidate.addElementInBlock(blockRef,*it);
	    }
	  active[*it] = false;	
	}      
      
      egCandidate_.push_back(temp_Candidate);
      egExtra_.push_back(myExtraEqual);
      
      return true;
      
//       bool mvaSelected = (BDToutput_[cgsf] >=  mvaEleCut_);
//       if( mvaSelected || bypassmva ) 	  {
// 	  elCandidate_.push_back(temp_Candidate);
// 	  if(itextra!=electronExtra_.end()) 
// 	    itextra->setStatus(PFCandidateElectronExtra::Selected,true);
// 	}
//       else 	  {
// 	if(itextra!=electronExtra_.end()) 
// 	  itextra->setStatus(PFCandidateElectronExtra::Rejected,true);
//       }
//       allElCandidate_.push_back(temp_Candidate);
//       
//       // save the status information
//       if(itextra!=electronExtra_.end()) {
// 	itextra->setStatus(PFCandidateElectronExtra::ECALDrivenPreselected,bypassmva);
// 	itextra->setStatus(PFCandidateElectronExtra::MVASelected,mvaSelected);
//       }
      

    }
    else {
      //BDToutput_[cgsf] = -1.;   // if the momentum is < 0.5 ID = false, but not sure
      // it could be misleading. 
      if( DebugIDCandidates ) 
	cout << "SetCandidates:: No Candidate Produced because of Pt cut: 0.5 " <<endl;
      return false;
    }
  } 
  else {
    //BDToutput_[cgsf] = -1.;  // if gsf ref does not exist
    if( DebugIDCandidates ) 
      cout << "SetCandidates:: No Candidate Produced because of No GSF Track Ref " <<endl;
    return false;
  }
  return false;
}

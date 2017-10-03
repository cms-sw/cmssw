#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFPhotonClusters.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFElectronExtraEqual.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CondFormats/ESObjects/interface/ESChannelStatus.h"

#include <TFile.h>
#include <TVector2.h>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <TMath.h>
#include "TMVA/MethodBDT.h"

// include combinations header (not yet included in boost)
#include "combination.hpp"

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
    bool _ispfsc;
    SeedMatchesToProtoObject(const reco::ElectronSeedRef& s) {
      _scfromseed = s->caloCluster().castTo<reco::SuperClusterRef>();
      _ispfsc = false;
      if( _scfromseed.isNonnull() ) {
	const edm::Ptr<reco::PFCluster> testCast(_scfromseed->seed());
	_ispfsc = testCast.isNonnull();
      }      
    }
    bool operator() (const PFEGammaAlgo::ProtoEGObject& po) {
      if( _scfromseed.isNull() || !po.parentSC ) return false;      
      if( _ispfsc ) {
	return ( _scfromseed->seed() == 
		 po.parentSC->superClusterRef()->seed() ); 
      }      
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
	  float cluster_e = elemasclus->clusterRef()->correctedEnergy();
	  float trk_pin   = elemasgsf->Pin().P();
	  if( cluster_e / trk_pin > EoPin_cut ) {
	    LOGDRESSED("elementNotCloserToOther")
	      << "GSF track failed EoP cut to match with cluster!";
	    return false;
	  }
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
	  float cluster_e = elemasclus->clusterRef()->correctedEnergy();
	  float trk_pin   = 
	    std::sqrt(elemaskf->trackRef()->innerMomentum().mag2());
	  if( cluster_e / trk_pin > EoPin_cut ) {
	    LOGDRESSED("elementNotCloserToOther")
	      << "KF track failed EoP cut to match with cluster!";
	    return false;
	  }
	}
      }	
      break;
    default:
      break;
    }	        

    const float dist = 
      block->dist(key,test,block->linkData(),reco::PFBlock::LINKTEST_ALL);
    if( dist == -1.0f ) return false; // don't associate non-linked elems
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
	    float cluster_e = elemasclus->clusterRef()->correctedEnergy();
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
	    float cluster_e = elemasclus->clusterRef()->correctedEnergy();
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
	LOGDRESSED("elementNotCloserToOther") 
	  << "key element of type " << keytype 
	  << " is closer to another element of type" << valtype 
	  << std::endl;
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
      const reco::PFClusterRef& cRef = elemascluster->clusterRef();
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
    // also don't allow ROs where both have clusters
    // and GSF tracks to merge (10 Dec 2013)
    if(!RO1.primaryGSFs.empty() && !RO2.primaryGSFs.empty()) {
      LOGDRESSED("isROLinkedByClusterOrTrack") 
	<< "cannot merge, both have GSFs!" << std::endl;
      return false;
    }
    // don't allow EB/EE to mix (11 Sept 2013)
    if( !RO1.ecalclusters.empty() && !RO2.ecalclusters.empty() ) {
      if(RO1.ecalclusters.front().first->clusterRef()->layer() !=
	 RO2.ecalclusters.front().first->clusterRef()->layer() ) {
	LOGDRESSED("isROLinkedByClusterOrTrack") 
	  << "cannot merge, different ECAL types!" << std::endl;
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
	if( not_closer ) {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	    << "merged by cluster to primary GSF" << std::endl;
	  return true;	  
	} else {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	    << "cluster to primary GSF failed since"
	    << " cluster closer to another GSF" << std::endl;
	}
      }
      for( const auto& primkf : RO2.primaryKFs) {
	not_closer = 
	  elementNotCloserToOther(blk,
				  cluster.first->type(),
				  cluster.first->index(),
				  primkf.first->type(),
				  primkf.first->index());
	if( not_closer ) {
	   LOGDRESSED("isROLinkedByClusterOrTrack") 
	     << "merged by cluster to primary KF" << std::endl;
	  return true;
	}
      }
      for( const auto& secdkf : RO2.secondaryKFs) {
	not_closer = 
	    elementNotCloserToOther(blk,
				    cluster.first->type(),
				    cluster.first->index(),
				    secdkf.first->type(),
				    secdkf.first->index());
	if( not_closer ) {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	     << "merged by cluster to secondary KF" << std::endl;
	  return true;	  
	}
      }
      // check links brem -> cluster
      for( const auto& brem : RO2.brems ) {
	not_closer = elementNotCloserToOther(blk,
					     cluster.first->type(),
					     cluster.first->index(),
					     brem.first->type(),
					     brem.first->index());
	if( not_closer ) {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	     << "merged by cluster to brem KF" << std::endl;
	  return true;
	}
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
	if( not_closer ) {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	     << "merged by GSF to secondary KF" << std::endl;
	  return true;	  
	}
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
	if( not_closer ) {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	     << "merged by primary KF to secondary KF" << std::endl;
	  return true;	  
	}
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
	if( not_closer ) {
	  LOGDRESSED("isROLinkedByClusterOrTrack") 
	     << "merged by secondary KF to secondary KF" << std::endl;
	  return true;	  
	}
      }
    }    
    return false;
  }
  
  struct TestIfROMergableByLink : public POMatcher {
    const PFEGammaAlgo::ProtoEGObject& comp;
    TestIfROMergableByLink(const PFEGammaAlgo::ProtoEGObject& RO) :
      comp(RO) {}
    bool operator() (const PFEGammaAlgo::ProtoEGObject& ro) {      
      const bool result = ( isROLinkedByClusterOrTrack(comp,ro) || 
			    isROLinkedByClusterOrTrack(ro,comp)   );      
      return result;      
    }
  }; 

  std::vector<const ClusterElement*> 
  getSCAssociatedECALsSafe(const reco::SuperClusterRef& scref,
			   std::vector<PFFlaggedElement>& ecals) {
    std::vector<const ClusterElement*> cluster_list;    
    auto sccl = scref->clustersBegin();
    auto scend = scref->clustersEnd();
    auto pfc = ecals.begin();
    auto pfcend = ecals.end();
    for( ; sccl != scend; ++sccl ) {
      std::vector<const ClusterElement*> matched_pfcs;
      const double eg_energy = (*sccl)->energy();

      for( pfc = ecals.begin(); pfc != pfcend; ++pfc ) {
	const ClusterElement *pfcel = 
	  docast(const ClusterElement*, pfc->first);
	const bool matched = 
	  ClusterClusterMapping::overlap(**sccl,*(pfcel->clusterRef()));
     // need to protect against high energy clusters being attached
     // to low-energy SCs          
	if( matched && pfcel->clusterRef()->energy() < 1.2*scref->energy()) {
	  matched_pfcs.push_back(pfcel);
	}
      }
      std::sort(matched_pfcs.begin(),matched_pfcs.end());

      double min_residual = 1e6;
      std::vector<const ClusterElement*> best_comb;
      for( size_t i = 1; i <= matched_pfcs.size(); ++i ) {
	//now we find the combination of PF-clusters which
	//has the smallest energy residual with respect to the
	//EG-cluster we are looking at now
	do {
	  double energy = std::accumulate(matched_pfcs.begin(),
					  matched_pfcs.begin()+i,
					  0.0,
					  [](const double a,
					     const ClusterElement* c) 
			       { return a + c->clusterRef()->energy(); });
	  const double resid = std::abs(energy - eg_energy);
	  if( resid < min_residual ) {
	    best_comb.clear();
	    best_comb.reserve(i);
	    min_residual = resid;
	    best_comb.insert(best_comb.begin(),
			     matched_pfcs.begin(),
			     matched_pfcs.begin()+i);
	  }
	}while(boost::next_combination(matched_pfcs.begin(),
				       matched_pfcs.begin()+i,
				       matched_pfcs.end()));	
      }
      for( const auto& clelem : best_comb ) {
	if( std::find(cluster_list.begin(),cluster_list.end(),clelem) ==
	    cluster_list.end() ) {
	  cluster_list.push_back(clelem);
	}
      }
    }
    return cluster_list;
  }
  bool addPFClusterToROSafe(const ClusterElement* cl,
			    PFEGammaAlgo::ProtoEGObject& RO) {
    if( RO.ecalclusters.empty() ) {
      RO.ecalclusters.push_back(std::make_pair(cl,true));      
      return true;
    } else {
      const PFLayer::Layer clayer = cl->clusterRef()->layer();
      const PFLayer::Layer blayer = 
	RO.ecalclusters.back().first->clusterRef()->layer();
      if( clayer == blayer ) {
	RO.ecalclusters.push_back(std::make_pair(cl,true));      
	return true;
      }
    }
    return false;
  }
  
  // sets the cluster best associated to the GSF track
  // leave it null if no GSF track
  void setROElectronCluster(PFEGammaAlgo::ProtoEGObject& RO) {
    if( RO.ecalclusters.empty() ) return;
    RO.lateBrem = -1;
    RO.firstBrem = -1;
    RO.nBremsWithClusters = -1;    
    const reco::PFBlockElementBrem *firstBrem = nullptr, *lastBrem = nullptr;
    const reco::PFBlockElementCluster *bremCluster = nullptr, *gsfCluster = nullptr,
      *kfCluster = nullptr, *gsfCluster_noassc = nullptr;
    const reco::PFBlockRef& parent = RO.parentBlock;
    int nBremClusters = 0;
    constexpr float maxDist = 1e6;
    float mDist_gsf(maxDist), mDist_gsf_noassc(maxDist), mDist_kf(maxDist);
    for( const auto& cluster : RO.ecalclusters ) {      
      for( const auto& gsf : RO.primaryGSFs ) {
	const bool hasclu = elementNotCloserToOther(parent,
						    gsf.first->type(),
						    gsf.first->index(),
						    cluster.first->type(),
						    cluster.first->index());
	const float deta = 
	  std::abs(cluster.first->clusterRef()->positionREP().eta() -
		   gsf.first->positionAtECALEntrance().eta());
	const float dphi = 
	  std::abs(TVector2::Phi_mpi_pi(
			cluster.first->clusterRef()->positionREP().phi() - 
			gsf.first->positionAtECALEntrance().phi()));
	const float dist = std::hypot(deta,dphi);
	if( hasclu && dist < mDist_gsf ) {	  
	  gsfCluster = cluster.first;
	  mDist_gsf = dist;
	} else if ( dist < mDist_gsf_noassc ) {
	  gsfCluster_noassc = cluster.first;
	  mDist_gsf_noassc = dist;
	}
      }    
      for( const auto& kf  : RO.primaryKFs ) {
	const bool hasclu = elementNotCloserToOther(parent,
						    kf.first->type(),
						    kf.first->index(),
						    cluster.first->type(),
						    cluster.first->index());
	const float dist = parent->dist(cluster.first->index(),
					kf.first->index(),
					parent->linkData(),
					reco::PFBlock::LINKTEST_ALL);
	if( hasclu && dist < mDist_kf ) {
	  kfCluster = cluster.first;
	  mDist_kf = dist;
	}
      }
      for( const auto& brem : RO.brems ) {
	const bool hasclu = elementNotCloserToOther(parent,
						    brem.first->type(),
						    brem.first->index(),
						    cluster.first->type(),
						    cluster.first->index());
	if( hasclu ) {	
	  ++nBremClusters;
	  if( !firstBrem || 
	      ( firstBrem->indTrajPoint() - 2 > 
		brem.first->indTrajPoint() - 2) ) {
	    firstBrem = brem.first;
	  }
	  if( !lastBrem || 
	      ( lastBrem->indTrajPoint() - 2 < 
		brem.first->indTrajPoint() - 2) ) {
	    lastBrem = brem.first;
	    bremCluster = cluster.first;
	  }
	}
      }
    }
    if( !gsfCluster && !kfCluster && !bremCluster ) {
      gsfCluster = gsfCluster_noassc;
    }
    RO.nBremsWithClusters = nBremClusters;
    RO.lateBrem = 0;
    if( gsfCluster ) {
      RO.electronClusters.push_back(gsfCluster);
    } else if ( kfCluster ) {
      RO.electronClusters.push_back(kfCluster);
    }
    if( bremCluster && !gsfCluster && !kfCluster ) {
      RO.electronClusters.push_back(bremCluster);
    }
    if( firstBrem && RO.ecalclusters.size() > 1 ) {      
      RO.firstBrem = firstBrem->indTrajPoint() - 2;
      if( bremCluster == gsfCluster ) RO.lateBrem = 1;
    }
  }
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
  excluded_(0.0), Mustache_EtRatio_(0.0), Mustache_Et_out_(0.0),
  channelStatus_(nullptr)
{   
  //Material Map
  TFile *XO_File = new TFile(cfg_.X0_Map.c_str(),"READ");
  X0_sum    = (TH2D*)XO_File->Get("TrackerSum");
  X0_inner  = (TH2D*)XO_File->Get("Inner");
  X0_middle = (TH2D*)XO_File->Get("Middle");
  X0_outer  = (TH2D*)XO_File->Get("Outer");
  
}

void PFEGammaAlgo::RunPFEG(const pfEGHelpers::HeavyObjectCache* hoc,
                           const reco::PFBlockRef&  blockRef,
                           std::vector<bool>& active) {  

  fifthStepKfTrack_.clear();
  convGsfTrack_.clear();
  
  egCandidate_.clear();
  egExtra_.clear();
 
  // define how much is printed out for debugging.
  // ... will be setable via CFG file parameter
  verbosityLevel_ = Chatty;          // Chatty mode.
  
  buildAndRefineEGObjects(hoc, blockRef);
}

float PFEGammaAlgo::
EvaluateSingleLegMVA(const pfEGHelpers::HeavyObjectCache* hoc,
                     const reco::PFBlockRef& blockref, 
                     const reco::Vertex& primaryvtx, 
                     unsigned int track_index) {  
  const reco::PFBlock& block = *blockref;  
  const edm::OwnVector< reco::PFBlockElement >& elements = block.elements();  
  //use this to store linkdata in the associatedElements function below  
  const PFBlock::LinkData& linkData =  block.linkData();  
  //calculate MVA Variables  
  chi2=elements[track_index].trackRef()->chi2()/elements[track_index].trackRef()->ndof(); 
  nlost=elements[track_index].trackRef()->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS); 
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
  if(!ecalAssoTrack.empty()) {  
    for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoTrack.begin();  
	itecal != ecalAssoTrack.end(); ++itecal) {  
      linked_e=linked_e+elements[itecal->second].clusterRef()->energy();  
    }  
  }  
  if(!hcalAssoTrack.empty()) {  
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
  
  float vars[] = { del_phi, nlayers, chi2, EoverPt,
                   HoverPt, track_pt, STIP, nlost };

  mvaValue = hoc->gbrSingleLeg_->GetAdaBoostClassifier(vars);
  
  return mvaValue;
}

bool PFEGammaAlgo::isAMuon(const reco::PFBlockElement& pfbe) {
  switch( pfbe.type() ) {
  case reco::PFBlockElement::GSF:    
    {
      auto& elements = _currentblock->elements();
      std::multimap<double,unsigned> tks;
      _currentblock->associatedElements(pfbe.index(),
					_currentlinks,
					tks,
					reco::PFBlockElement::TRACK,
					reco::PFBlock::LINKTEST_ALL);      
      for( const auto& tk : tks ) {
	if( PFMuonAlgo::isMuon(elements[tk.second]) ) {
	  return true;
	}
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

void PFEGammaAlgo::buildAndRefineEGObjects(const pfEGHelpers::HeavyObjectCache* hoc,
                                           const reco::PFBlockRef& block) {
  LOGVERB("PFEGammaAlgo") 
    << "Resetting PFEGammaAlgo for new block and running!" << std::endl;
  _splayedblock.clear();
  _recoveredlinks.clear();
  _refinableObjects.clear();
  _finalCandidates.clear();  
  _splayedblock.resize(13); // make sure that we always have the HGCAL entry

  _currentblock = block;
  _currentlinks = block->linkData();
  //LOGDRESSED("PFEGammaAlgo") << *_currentblock << std::endl;
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
    // do the same for HCAL clusters associated to the GSF
    linkRefinableObjectPrimaryGSFTrackToHCAL(RO);
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
    linkRefinableObjectECALToSingleLegConv(hoc,RO);
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

  // -- unlinking and proto-object vetos, final sorting
  for( auto& RO : _refinableObjects ) {
    // remove secondary KFs (and possibly ECALs) matched to HCAL clusters
    unlinkRefinableObjectKFandECALMatchedToHCAL(RO, false, false);
    // remove secondary KFs and ECALs linked to them that have bad E/p_in 
    // and spoil the resolution
    unlinkRefinableObjectKFandECALWithBadEoverP(RO);
    // put things back in order after partitioning
    std::sort(RO.ecalclusters.begin(), RO.ecalclusters.end(),
	    [](const PFClusterFlaggedElement& a,
	       const PFClusterFlaggedElement& b) 
	    { return ( a.first->clusterRef()->correctedEnergy() > 
		       b.first->clusterRef()->correctedEnergy() ) ; });
    setROElectronCluster(RO);
  }

  LOGDRESSED("PFEGammaAlgo")
    << "There are " << _refinableObjects.size() 
    << " after the unlinking and vetos step." << std::endl;
  dumpCurrentRefinableObjects();

  // fill the PF candidates and then build the refined SC
  fillPFCandidates(hoc,_refinableObjects,outcands_,outcandsextra_);

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
    fromSC.nBremsWithClusters = -1;
    fromSC.firstBrem = -1;
    fromSC.lateBrem = -1;
    fromSC.parentBlock = _currentblock;
    fromSC.parentSC = docast(const PFSCElement*,element.first);
    // splay the supercluster so we can knock out used elements
    bool sc_success = 
      unwrapSuperCluster(fromSC.parentSC,fromSC.ecalclusters,fromSC.ecal2ps);
    if( sc_success ) {
      /*
      auto ins_pos = std::lower_bound(_refinableObjects.begin(),
				      _refinableObjects.end(),
				      fromSC,
				      [&](const ProtoEGObject& a,
					  const ProtoEGObject& b){
					const double a_en = 
					a.parentSC->superClusterRef()->energy();
					const double b_en = 
					b.parentSC->superClusterRef()->energy();
					return a_en < b_en;
				      });
      */
      _refinableObjects.insert(_refinableObjects.end(),fromSC);
    }
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
    fromGSF.nBremsWithClusters = -1;
    fromGSF.firstBrem = -1;
    fromGSF.lateBrem = 0;
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
	 LOGDRESSED("PFEGammaAlgo")
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
     /*
     auto ins_pos = std::lower_bound(_refinableObjects.begin(),
				     _refinableObjects.end(),
				     fromGSF,
				     [&](const ProtoEGObject& a,
					 const ProtoEGObject& b){
				       const double a_en = ( a.parentSC ?
							     a.parentSC->superClusterRef()->energy() :
							     a.primaryGSFs[0].first->GsftrackRef()->pt() );
				       const double b_en = ( b.parentSC ?
							     b.parentSC->superClusterRef()->energy() :
							     b.primaryGSFs[0].first->GsftrackRef()->pt() );
				       return a_en < b_en;
				     });   
     */
     _refinableObjects.insert(_refinableObjects.end(),fromGSF);
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
   auto hgcalbegin = _splayedblock[reco::PFBlockElement::HGCAL].begin();
   auto hgcalend = _splayedblock[reco::PFBlockElement::HGCAL].end(); 
   if( ecalbegin == ecalend && hgcalbegin == hgcalend ) {
     LOGERR("PFEGammaAlgo::unwrapSuperCluster()")
       << "There are no ECAL elements in a block with imported SC!" 
       << " This is a bug we should fix this!" 
       << std::endl;
     return false;
   }
   reco::SuperClusterRef scref = thesc->superClusterRef();
   const bool is_pf_sc = thesc->fromPFSuperCluster();
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
   NotCloserToOther<reco::PFBlockElement::SC,reco::PFBlockElement::HGCAL> 
     hgcalClustersInSC(_currentblock,_currentlinks,thesc);
   auto ecalfirstnotinsc = std::partition(ecalbegin,ecalend,ecalClustersInSC);
   auto hgcalfirstnotinsc = std::partition(hgcalbegin,hgcalend,hgcalClustersInSC);
   //reset the begin and end iterators
   ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
   ecalend = _splayedblock[reco::PFBlockElement::ECAL].end();  

   hgcalbegin = _splayedblock[reco::PFBlockElement::HGCAL].begin();
   hgcalend = _splayedblock[reco::PFBlockElement::HGCAL].end();  

   //get list of associated clusters by det id and energy matching
   //(only needed when using non-pf supercluster)
   std::vector<const ClusterElement*> safePFClusters = is_pf_sc ? std::vector<const ClusterElement*>() : getSCAssociatedECALsSafe(scref,_splayedblock[reco::PFBlockElement::ECAL]);
   
   if( ecalfirstnotinsc == ecalbegin &&  
       hgcalfirstnotinsc == hgcalbegin) {
     LOGERR("PFEGammaAlgo::unwrapSuperCluster()")
       << "No associated block elements to SuperCluster!" 
       << " This is a bug we should fix!"
       << std::endl;
     return false;
   }
   npfclusters = std::distance(ecalbegin,ecalfirstnotinsc) + std::distance(hgcalbegin,hgcalfirstnotinsc);
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
   for( auto ecalitr = ecalbegin; ecalitr != ecalfirstnotinsc; ++ecalitr ) {    
     const PFClusterElement* elemascluster = 
       docast(const PFClusterElement*,ecalitr->first);

     // reject clusters that really shouldn't be associated to the SC
     // (only needed when using non-pf-supercluster)
     if(!is_pf_sc && std::find(safePFClusters.begin(),safePFClusters.end(),elemascluster) ==
	safePFClusters.end() ) continue;

     //add cluster
     ecalclusters.push_back(std::make_pair(elemascluster,true));
     //mark cluster as used
     ecalitr->second = false;     
     
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
     npfpsclusters += attachPSClusters(elemascluster,eslist);    
   } // loop over ecal elements

   for( auto hgcalitr = hgcalbegin; hgcalitr != hgcalfirstnotinsc; ++hgcalitr ) {    
     const PFClusterElement* elemascluster = 
       docast(const PFClusterElement*,hgcalitr->first);

     // reject clusters that really shouldn't be associated to the SC
     // (only needed when using non-pf-supercluster)
     if(!is_pf_sc && std::find(safePFClusters.begin(),safePFClusters.end(),elemascluster) ==
	safePFClusters.end() ) continue;

     //add cluster
     ecalclusters.push_back(std::make_pair(elemascluster,true));
     //mark cluster as used
     hgcalitr->second = false;     
   } // loop over ecal elements
   
   /*
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
   */

   LOGDRESSED("PFEGammaAlgo")
     << " Unwrapped SC has " << npfclusters << " ECAL sub-clusters"
     << " and " << npfpsclusters << " PreShower layers 1 & 2 clusters!" 
     << std::endl; 
   return true;
 }



 int PFEGammaAlgo::attachPSClusters(const ClusterElement* ecalclus,
				    ClusterMap::mapped_type& eslist) {  
   if( ecalclus->clusterRef()->layer() == PFLayer::ECAL_BARREL ) return 0;
   edm::Ptr<reco::PFCluster> clusptr = refToPtr(ecalclus->clusterRef());
   EEtoPSElement ecalkey = std::make_pair(clusptr.key(),clusptr);
   auto assc_ps = std::equal_range(eetops_->cbegin(),
				   eetops_->cend(),
				   ecalkey,
				   comparePSMapByKey);
   for( const auto& ps1 : _splayedblock[reco::PFBlockElement::PS1] ) {
     edm::Ptr<reco::PFCluster> temp = refToPtr(ps1.first->clusterRef());
     for( auto pscl = assc_ps.first; pscl != assc_ps.second; ++pscl ) {
       if( pscl->second == temp ) {
	 const ClusterElement* pstemp = 
	   docast(const ClusterElement*,ps1.first);
	 eslist.push_back( PFClusterFlaggedElement(pstemp,true) );
       }
     }
   }
   for( const auto& ps2 : _splayedblock[reco::PFBlockElement::PS2] ) {
     edm::Ptr<reco::PFCluster> temp = refToPtr(ps2.first->clusterRef());
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
       info << "firstBrem : " << ro.firstBrem 
	    << " lateBrem : " << ro.lateBrem
	    << " nBrems with cluster : " << ro.nBremsWithClusters
	    << std::endl;;
       if( ro.electronClusters.size() && ro.electronClusters[0] ) {
	 info << "electron cluster : ";
	 ro.electronClusters[0]->Dump(info,"\t");
	 info << std::endl;
       } else {
	 info << " no electron cluster." << std::endl;
       }	 
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
   typedef std::multimap<double, unsigned> MatchedMap;
   typedef const reco::PFBlockElementGsfTrack* GsfTrackElementPtr;
   if( _splayedblock[reco::PFBlockElement::ECAL].empty() ||
       _splayedblock[reco::PFBlockElement::TRACK].empty()   ) return;
   MatchedMap matchedGSFs, matchedECALs;
   std::unordered_map<GsfTrackElementPtr,MatchedMap> gsf_ecal_cache;
   for( auto& kftrack : _splayedblock[reco::PFBlockElement::TRACK] ) {
     matchedGSFs.clear();
     _currentblock->associatedElements(kftrack.first->index(), _currentlinks,
				       matchedGSFs,
				       reco::PFBlockElement::GSF,
				       reco::PFBlock::LINKTEST_ALL);
     if( matchedGSFs.empty() ) { // only run this if we aren't associated to GSF
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
	   // make sure cache exists
	   if( !gsf_ecal_cache.count(elemasgsf) ) {
	     matchedECALs.clear();
	     _currentblock->associatedElements(elemasgsf->index(), _currentlinks,
					       matchedECALs,
					       reco::PFBlockElement::ECAL,
					       reco::PFBlock::LINKTEST_ALL);
	     gsf_ecal_cache.emplace(elemasgsf,matchedECALs);
	     MatchedMap().swap(matchedECALs);
	   } 
	   const MatchedMap& ecal_matches = gsf_ecal_cache[elemasgsf];	   
	   if( !ecal_matches.empty() ) {
	     if( ecal_matches.begin()->second == closestECAL.first->index() ) {
	       gsflinked = true;
	       break;
	     }
	   }			    
	 } // loop over primary GSF tracks
	 if( !gsflinked && !inSC) { 
	   // determine if we should remove the matched cluster
	   const reco::PFBlockElementTrack * kfEle = 
	     docast(const reco::PFBlockElementTrack*,kftrack.first);
	   const reco::TrackRef& trackref = kfEle->trackRef();

	   const int nexhits = 
	     trackref->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
	   bool fromprimaryvertex = false;
	   for( auto vtxtks = cfg_.primaryVtx->tracks_begin();
		vtxtks != cfg_.primaryVtx->tracks_end(); ++ vtxtks ) {
	     if( trackref == vtxtks->castTo<reco::TrackRef>() ) {
	       fromprimaryvertex = true;
	       break;
	     }
	   }// loop over tracks in primary vertex
	    // if associated to good non-GSF matched track remove this cluster
	   if( PFTrackAlgoTools::isGoodForEGMPrimary(trackref->algo()) && nexhits == 0 && fromprimaryvertex ) {
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
     // bugfix for early termination merging loop (15 April 2014)
     // check all pairwise combinations in the list
     // if one has a merge shuffle it to the front of the list
     // if there are no merges left to do we can terminate
     for( auto it1 = ROs.begin(); it1 != ROs.end(); ++it1 ) {
       TestIfROMergableByLink mergeTest(*it1);
       auto find_start = it1; ++find_start;
       auto has_merge = std::find_if(find_start,ROs.end(),mergeTest);
       if( has_merge != ROs.end() && it1 != ROs.begin() ) {
	 std::swap(*(ROs.begin()),*it1);
	 break;
       }
     }// ensure mergables are shuffled to the front
     ProtoEGObject& thefront = ROs.front();
     TestIfROMergableByLink mergeTest(thefront);
     auto mergestart = ROs.begin(); ++mergestart;    
     auto nomerge = std::partition(mergestart,ROs.end(),mergeTest);
     if( nomerge != mergestart ) {
       LOGDRESSED("PFEGammaAlgo::mergeROsByAnyLink()")       
	 << "Found objects " << std::distance(mergestart,nomerge)
	 << " to merge by links to the front!" << std::endl;
       for( auto roToMerge = mergestart; roToMerge != nomerge; ++roToMerge) {
         //bugfix! L.Gray 14 Jan 2016 
         // -- check that the front is still mergeable!
         if( !thefront.ecalclusters.empty() && !roToMerge->ecalclusters.empty() ) {
           if( thefront.ecalclusters.front().first->clusterRef()->layer() !=   
               roToMerge->ecalclusters.front().first->clusterRef()->layer() ) {
             LOGWARN("PFEGammaAlgo::mergeROsByAnyLink") 
               << "Tried to merge EB and EE clusters! Skipping!";
             ROs.push_back(*roToMerge);
             continue;
           }
         }         
         //end bugfix
	 thefront.ecalclusters.insert(thefront.ecalclusters.end(),
				      roToMerge->ecalclusters.begin(),
				      roToMerge->ecalclusters.end());
	 thefront.ecal2ps.insert(roToMerge->ecal2ps.begin(),
				 roToMerge->ecal2ps.end());
	 thefront.secondaryKFs.insert(thefront.secondaryKFs.end(),
				      roToMerge->secondaryKFs.begin(),
				      roToMerge->secondaryKFs.end());

	 thefront.localMap.insert(thefront.localMap.end(),
				  roToMerge->localMap.begin(),
				  roToMerge->localMap.end());
	 // TO FIX -> use best (E_gsf - E_clustersum)/E_GSF
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
	   thefront.electronClusters = roToMerge->electronClusters;
	   thefront.nBremsWithClusters = roToMerge->nBremsWithClusters;
	   thefront.firstBrem = roToMerge->firstBrem;
	   thefront.lateBrem = roToMerge->lateBrem;
	 } else if ( thefront.electronSeed.isNonnull() && 
		     roToMerge->electronSeed.isNonnull()) {
	   LOGWARN("PFEGammaAlgo::mergeROsByAnyLink")
	     << "Need to implement proper merging of two gsf candidates!"
	     << std::endl;
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
  if( _splayedblock[reco::PFBlockElement::TRACK].empty() ) return;
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
  if( _splayedblock[reco::PFBlockElement::TRACK].empty() ) return;
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
  if( _splayedblock[reco::PFBlockElement::ECAL].empty() ) {
    RO.electronClusters.push_back(nullptr);
    return; 
  }
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
  for( auto& primgsf : RO.primaryGSFs ) {    
    NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::ECAL>
      gsfTracksToECALs(_currentblock,_currentlinks,primgsf.first);
    CompatibleEoPOut eoverp_test(primgsf.first);
    // get set of matching ecals not already in SC
    auto notmatched_blk = std::partition(ECALbegin,ECALend,gsfTracksToECALs);
    notmatched_blk = std::partition(ECALbegin,notmatched_blk,eoverp_test);
    // get set of matching ecals already in the RO
    auto notmatched_sc = std::partition(RO.ecalclusters.begin(),
					RO.ecalclusters.end(),
					gsfTracksToECALs);
    notmatched_sc = std::partition(RO.ecalclusters.begin(),
				   notmatched_sc,
				   eoverp_test);
    // look inside the SC for the ECAL cluster
    for( auto ecal = RO.ecalclusters.begin(); ecal != notmatched_sc; ++ecal ) {
      const PFClusterElement* elemascluster = 
	docast(const PFClusterElement*,ecal->first);    
      PFClusterFlaggedElement temp(elemascluster,true);
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") 
	<< "Found a cluster already in RO by GSF extrapolation"
	<< " at ECAL surface!" << std::endl
	<< *elemascluster << std::endl;
            
      RO.localMap.push_back(ElementMap::value_type(primgsf.first,temp.first));
      RO.localMap.push_back(ElementMap::value_type(temp.first,primgsf.first));
    }
    // look outside the SC for the ecal cluster
    for( auto ecal = ECALbegin; ecal != notmatched_blk; ++ecal ) {
      const PFClusterElement* elemascluster = 
	docast(const PFClusterElement*,ecal->first);    
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") 
	<< "Found a cluster not already in RO by GSF extrapolation"
	<< " at ECAL surface!" << std::endl
	<< *elemascluster << std::endl;
      if( addPFClusterToROSafe(elemascluster,RO) ) {
	attachPSClusters(elemascluster,RO.ecal2ps[elemascluster]);      
	RO.localMap.push_back(ElementMap::value_type(primgsf.first,elemascluster));
	RO.localMap.push_back(ElementMap::value_type(elemascluster,primgsf.first));
	ecal->second = false;    
      }
    }    
  }
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::
linkRefinableObjectPrimaryGSFTrackToHCAL(ProtoEGObject& RO) {
  if( _splayedblock[reco::PFBlockElement::HCAL].empty() ) return; 
  auto HCALbegin = _splayedblock[reco::PFBlockElement::HCAL].begin();
  auto HCALend = _splayedblock[reco::PFBlockElement::HCAL].end();
  for( auto& primgsf : RO.primaryGSFs ) {
    NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::HCAL>
      gsfTracksToHCALs(_currentblock,_currentlinks,primgsf.first);
    CompatibleEoPOut eoverp_test(primgsf.first);
    auto notmatched = std::partition(HCALbegin,HCALend,gsfTracksToHCALs);    
    for( auto hcal = HCALbegin; hcal != notmatched; ++hcal ) { 
      const PFClusterElement* elemascluster = 
	docast(const PFClusterElement*,hcal->first);    
      PFClusterFlaggedElement temp(elemascluster,true);    
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") 
	<< "Found an HCAL cluster associated to GSF extrapolation" 
	<< std::endl;
      RO.hcalClusters.push_back(temp);
      RO.localMap.push_back( ElementMap::value_type(primgsf.first,temp.first) );
      RO.localMap.push_back( ElementMap::value_type(temp.first,primgsf.first) );
      hcal->second = false;
    }
  }
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::
linkRefinableObjectKFTracksToECAL(ProtoEGObject& RO) {
  if( _splayedblock[reco::PFBlockElement::ECAL].empty() ) return;  
  for( auto& primkf : RO.primaryKFs ) linkKFTrackToECAL(primkf,RO);
  for( auto& secdkf : RO.secondaryKFs ) linkKFTrackToECAL(secdkf,RO);
}

void 
PFEGammaAlgo::linkKFTrackToECAL(const KFFlaggedElement& kfflagged,
				ProtoEGObject& RO) {
  std::vector<PFClusterFlaggedElement>& currentECAL = RO.ecalclusters;
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();  
  NotCloserToOther<reco::PFBlockElement::TRACK,reco::PFBlockElement::ECAL>
    kfTrackToECALs(_currentblock,_currentlinks,kfflagged.first);      
  NotCloserToOther<reco::PFBlockElement::GSF,reco::PFBlockElement::ECAL>
    kfTrackGSFToECALs(_currentblock,_currentlinks,kfflagged.first);
  //get the ECAL elements not used and not closer to another KF
  auto notmatched_sc = std::partition(currentECAL.begin(),
				      currentECAL.end(),
				      kfTrackToECALs);
  //get subset ECAL elements not used or closer to another GSF of any type
  notmatched_sc = std::partition(currentECAL.begin(),
				 notmatched_sc,
				 kfTrackGSFToECALs);
  for( auto ecalitr = currentECAL.begin(); ecalitr != notmatched_sc; 
       ++ecalitr ) {
    const PFClusterElement* elemascluster = 
      docast(const PFClusterElement*,ecalitr->first);
    PFClusterFlaggedElement flaggedclus(elemascluster,true);
        
    LOGDRESSED("PFEGammaAlgo::linkKFTracktoECAL()") 
	<< "Found a cluster already in RO by KF extrapolation"
	<< " at ECAL surface!" << std::endl
	<< *elemascluster << std::endl;
    RO.localMap.push_back(ElementMap::value_type(elemascluster,
						 kfflagged.first));
    RO.localMap.push_back(ElementMap::value_type(kfflagged.first,
						 elemascluster));
  }
  //get the ECAL elements not used and not closer to another KF
  auto notmatched_blk = std::partition(ECALbegin,ECALend,kfTrackToECALs);
  //get subset ECAL elements not used or closer to another GSF of any type
  notmatched_blk = std::partition(ECALbegin,notmatched_blk,kfTrackGSFToECALs);
  for( auto ecalitr = ECALbegin; ecalitr != notmatched_blk; ++ecalitr ) {
    const PFClusterElement* elemascluster = 
      docast(const PFClusterElement*,ecalitr->first);
    if( addPFClusterToROSafe(elemascluster,RO) ) {
      attachPSClusters(elemascluster,RO.ecal2ps[elemascluster]);	  
      ecalitr->second = false;
      
      LOGDRESSED("PFEGammaAlgo::linkKFTracktoECAL()") 
	<< "Found a cluster not in RO by KF extrapolation"
	<< " at ECAL surface!" << std::endl
	<< *elemascluster << std::endl;
      RO.localMap.push_back(ElementMap::value_type(elemascluster,
						   kfflagged.first));
      RO.localMap.push_back( ElementMap::value_type(kfflagged.first,
						    elemascluster));
    }
  }  
}

void PFEGammaAlgo::
linkRefinableObjectBremTangentsToECAL(ProtoEGObject& RO) {
  if( RO.brems.empty() ) return;
  int FirstBrem = -1;
  int TrajPos = -1;
  int lastBremTrajPos = -1;  
  for( auto& bremflagged : RO.brems ) {
    bool has_clusters = false;
    TrajPos = (bremflagged.first->indTrajPoint())-2;
    auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
    auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
    NotCloserToOther<reco::PFBlockElement::BREM,reco::PFBlockElement::ECAL>
      BremToECALs(_currentblock,_currentlinks,bremflagged.first);
    // check for late brem using clusters already in the SC
    auto RSCBegin = RO.ecalclusters.begin();
    auto RSCEnd = RO.ecalclusters.end();
    auto notmatched_rsc = std::partition(RSCBegin,RSCEnd,BremToECALs);
    for( auto ecal = RSCBegin; ecal != notmatched_rsc; ++ecal ) {
      float deta = 
	std::abs( ecal->first->clusterRef()->positionREP().eta() -
		  bremflagged.first->positionAtECALEntrance().eta() );
      if( deta < 0.015 ) {
	has_clusters = true;
	if( lastBremTrajPos == -1 || lastBremTrajPos < TrajPos ) {
	  lastBremTrajPos = TrajPos;	  
	}
	if( FirstBrem == -1 || TrajPos < FirstBrem ) { // set brem information
	  FirstBrem = TrajPos;
	  RO.firstBrem = TrajPos;
	}	
	LOGDRESSED("PFEGammaAlgo::linkBremToECAL()") 
	  << "Found a cluster already in SC linked to brem extrapolation"
	  << " at ECAL surface!" << std::endl;
	RO.localMap.push_back( ElementMap::value_type(ecal->first,bremflagged.first) );
	RO.localMap.push_back( ElementMap::value_type(bremflagged.first,ecal->first) );
      }
    }
    // grab new clusters from the block (ensured to not be late brem)
    auto notmatched_block = std::partition(ECALbegin,ECALend,BremToECALs);   
    for( auto ecal = ECALbegin; ecal != notmatched_block; ++ecal ) {
      float deta = 
	std::abs( ecal->first->clusterRef()->positionREP().eta() -
		  bremflagged.first->positionAtECALEntrance().eta() );
      if( deta < 0.015 ) { 	
	has_clusters = true;
	if( lastBremTrajPos == -1 || lastBremTrajPos < TrajPos ) {
	  lastBremTrajPos = TrajPos;	  
	}
	if( FirstBrem == -1 || TrajPos < FirstBrem ) { // set brem information
	  
	  FirstBrem = TrajPos;
	  RO.firstBrem = TrajPos;
	}	
	const PFClusterElement* elemasclus =
	  docast(const PFClusterElement*,ecal->first);    
	if( addPFClusterToROSafe(elemasclus,RO) ) {
	  attachPSClusters(elemasclus,RO.ecal2ps[elemasclus]);
	  
	  RO.localMap.push_back( ElementMap::value_type(ecal->first,bremflagged.first) );
	  RO.localMap.push_back( ElementMap::value_type(bremflagged.first,ecal->first) );
	  ecal->second = false;
	  LOGDRESSED("PFEGammaAlgo::linkBremToECAL()") 
	    << "Found a cluster not already associated by brem extrapolation"
	    << " at ECAL surface!" << std::endl;
	}
	
      }
    }
    if(has_clusters) {
      if( RO.nBremsWithClusters == -1 ) RO.nBremsWithClusters = 0;
      ++RO.nBremsWithClusters;
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
    const std::vector<PFKFFlaggedElement>& secKFs = RO.secondaryKFs; //we want the entry at the index but we allocate to secondaryKFs in loop which invalidates all iterators, references and pointers, hence we need to get the entry fresh each time
    NotCloserToOther<reco::PFBlockElement::TRACK,
                     reco::PFBlockElement::TRACK,
                     true> 
      TracksToTracks(_currentblock,_currentlinks, secKFs[idx].first); 
    auto notmatched = std::partition(KFbegin,KFend,TracksToTracks);    
    notmatched = std::partition(KFbegin,notmatched,isConvKf);    
    for( auto kf = KFbegin; kf != notmatched; ++kf ) {
      const reco::PFBlockElementTrack* elemaskf =
	docast(const reco::PFBlockElementTrack*,kf->first);      
      RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
      RO.localMap.push_back( ElementMap::value_type(secKFs[idx].first,kf->first) );
      RO.localMap.push_back( ElementMap::value_type(kf->first,secKFs[idx].first) );
      kf->second = false;      
    }    
  }
}

void PFEGammaAlgo::
linkRefinableObjectECALToSingleLegConv(const pfEGHelpers::HeavyObjectCache* hoc,
                                       ProtoEGObject& RO) { 
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
      float mvaval = EvaluateSingleLegMVA(hoc,_currentblock, 
                                          *cfg_.primaryVtx, 
                                          kf->first->index());
      if(mvaval > cfg_.mvaConvCut) {
	const reco::PFBlockElementTrack* elemaskf =
	  docast(const reco::PFBlockElementTrack*,kf->first);
	RO.secondaryKFs.push_back( std::make_pair(elemaskf,true) );
	RO.localMap.push_back( ElementMap::value_type(ecal.first,elemaskf) );
	RO.localMap.push_back( ElementMap::value_type(elemaskf,ecal.first) );
	kf->second = false;
        
        RO.singleLegConversionMvaMap.insert(std::make_pair(elemaskf, mvaval));
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
      if( addPFClusterToROSafe(elemascluster,RO) ) {
	attachPSClusters(elemascluster,RO.ecal2ps[elemascluster]);
	RO.localMap.push_back(ElementMap::value_type(skf.first,elemascluster));
	RO.localMap.push_back(ElementMap::value_type(elemascluster,skf.first));
	ecal->second = false;      
      }
    }
  }
}

void PFEGammaAlgo::
fillPFCandidates(const pfEGHelpers::HeavyObjectCache* hoc,
                 const std::list<PFEGammaAlgo::ProtoEGObject>& ROs,
		 reco::PFCandidateCollection& egcands,
		 reco::PFCandidateEGammaExtraCollection& egxs) {
  // reset output collections
  egcands.clear();
  egxs.clear();  
  refinedscs_.clear();
  egcands.reserve(ROs.size());
  egxs.reserve(ROs.size());
  refinedscs_.reserve(ROs.size());
  for( auto& RO : ROs ) {    
    if( RO.ecalclusters.empty()  && 
	!cfg_.produceEGCandsWithNoSuperCluster ) continue;
    
    reco::PFCandidate cand;
    reco::PFCandidateEGammaExtra xtra;
    if( !RO.primaryGSFs.empty() || !RO.primaryKFs.empty() ) {
      cand.setPdgId(-11); // anything with a primary track is an electron
    } else {
      cand.setPdgId(22); // anything with no primary track is a photon
    }    
    if( !RO.primaryKFs.empty() ) {
      cand.setCharge(RO.primaryKFs[0].first->trackRef()->charge());
      xtra.setKfTrackRef(RO.primaryKFs[0].first->trackRef());
      cand.setTrackRef(RO.primaryKFs[0].first->trackRef());
      cand.addElementInBlock(_currentblock,RO.primaryKFs[0].first->index());
    }
    if( !RO.primaryGSFs.empty() ) {        
      cand.setCharge(RO.primaryGSFs[0].first->GsftrackRef()->chargeMode());
      xtra.setGsfTrackRef(RO.primaryGSFs[0].first->GsftrackRef());
      cand.setGsfTrackRef(RO.primaryGSFs[0].first->GsftrackRef());
      cand.addElementInBlock(_currentblock,RO.primaryGSFs[0].first->index());
    }
    if( RO.parentSC ) {
      xtra.setSuperClusterPFECALRef(RO.parentSC->superClusterRef());      
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
      if( RO.ecal2ps.count(clus) ) {
	for( auto& ps : RO.ecal2ps.at(clus) ) {
	  const PFClusterElement* psclus = ps.first;
	  cand.addElementInBlock(_currentblock,psclus->index());	
	}
      }
    }
    // add secondary tracks
    for( const auto& secdkf : RO.secondaryKFs ) {
      const PFKFElement* kf = secdkf.first;
      cand.addElementInBlock(_currentblock,kf->index());
      const reco::ConversionRefVector& convrefs = kf->convRefs();
      bool no_conv_ref = true;
      for( const auto& convref : convrefs ) {
	if( convref.isNonnull() && convref.isAvailable() ) {
	  xtra.addConversionRef(convref);
	  no_conv_ref = false;
	}
      }
      if( no_conv_ref ) {
        //single leg conversions
        
        //look for stored mva value in map or else recompute
        const auto &mvavalmapped = RO.singleLegConversionMvaMap.find(kf);
        //FIXME: Abuse single mva value to store both provenance and single leg mva score
        //by storing 3.0 + mvaval
        float mvaval = ( mvavalmapped != RO.singleLegConversionMvaMap.end() ? 
                         mvavalmapped->second : 
                         3.0 + EvaluateSingleLegMVA(hoc,_currentblock,
                                                    *cfg_.primaryVtx, 
                                                    kf->index()) );
        
        xtra.addSingleLegConvTrackRefMva(std::make_pair(kf->trackRef(),mvaval));
      }
    }
    
    // build the refined supercluster from those clusters left in the cand
    refinedscs_.push_back(buildRefinedSuperCluster(RO));

    //*TODO* cluster time is not reliable at the moment, so only use track timing
    float trkTime = 0, trkTimeErr = -1;
    if (!RO.primaryGSFs.empty() && RO.primaryGSFs[0].first->isTimeValid()) {
        trkTime = RO.primaryGSFs[0].first->time();
        trkTimeErr = RO.primaryGSFs[0].first->timeError();
    } else if (!RO.primaryKFs.empty() && RO.primaryKFs[0].first->isTimeValid()) {
        trkTime = RO.primaryKFs[0].first->time();
        trkTimeErr = RO.primaryKFs[0].first->timeError();
    }
    if (trkTimeErr >= 0) {
      cand.setTime( trkTime, trkTimeErr );
    }
    
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
      cand.setEcalEnergy(the_sc.rawEnergy(),the_sc.energy());
    } else if ( cfg_.produceEGCandsWithNoSuperCluster && 
		!RO.primaryGSFs.empty() ) {
      const PFGSFElement* gsf = RO.primaryGSFs[0].first;
      const reco::GsfTrackRef& gref = gsf->GsftrackRef();
      math::XYZTLorentzVector p4(gref->pxMode(),gref->pyMode(),
				 gref->pzMode(),gref->pMode());
      cand.setP4(p4);      
      cand.setPositionAtECALEntrance(gsf->positionAtECALEntrance());
    } else if ( cfg_.produceEGCandsWithNoSuperCluster &&
		!RO.primaryKFs.empty() ) {
      const PFKFElement* kf = RO.primaryKFs[0].first;
      reco::TrackRef kref = RO.primaryKFs[0].first->trackRef();
      math::XYZTLorentzVector p4(kref->px(),kref->py(),kref->pz(),kref->p());
      cand.setP4(p4);   
      cand.setPositionAtECALEntrance(kf->positionAtECALEntrance());
    }    
    const float ele_mva_value = calculate_ele_mva(hoc,RO,xtra);
    fill_extra_info(RO,xtra);
    //std::cout << "PFEG ele_mva: " << ele_mva_value << std::endl;
    xtra.setMVA(ele_mva_value);    
    cand.set_mva_e_pi(ele_mva_value);
    egcands.push_back(cand);
    egxs.push_back(xtra);    
  }
}

float PFEGammaAlgo::
calculate_ele_mva(const pfEGHelpers::HeavyObjectCache* hoc,
                  const PFEGammaAlgo::ProtoEGObject& RO,
		  reco::PFCandidateEGammaExtra& xtra) {
  if( RO.primaryGSFs.empty() ) return -2.0f;
  const PFGSFElement* gsfElement = RO.primaryGSFs.front().first;
  const PFKFElement* kfElement = nullptr;
  if( !RO.primaryKFs.empty() ) kfElement = RO.primaryKFs.front().first;
  reco::GsfTrackRef RefGSF= gsfElement->GsftrackRef();
  reco::TrackRef RefKF;
  constexpr float m_el = 0.000511;
  const double Ein_gsf = std::hypot(RefGSF->pMode(),m_el);
  double deta_gsfecal = 1e6;
  double sigmaEtaEta = 1e-14;
  const double Ene_hcalgsf = std::accumulate(RO.hcalClusters.begin(),
					     RO.hcalClusters.end(),
					     0.0,
					[](const double a,
					   const PFClusterFlaggedElement& b) 
				{ return a + b.first->clusterRef()->energy(); }
					     );
  if( !RO.primaryKFs.empty() ) {
    RefKF = RO.primaryKFs.front().first->trackRef();
  }
  const double Eout_gsf = gsfElement->Pout().t();
  const double Etaout_gsf = gsfElement->positionAtECALEntrance().eta();
  double FirstEcalGsfEnergy(0.0), OtherEcalGsfEnergy(0.0), EcalBremEnergy(0.0);
  //shower shape of cluster closest to gsf track
  std::vector<const reco::PFCluster*> gsfcluster;  
  for( const auto& ecal : RO.ecalclusters ) {
    const double cenergy = ecal.first->clusterRef()->correctedEnergy();
    ElementMap::value_type gsfToEcal(gsfElement,ecal.first);
    ElementMap::value_type kfToEcal(kfElement,ecal.first);
    bool hasgsf = 
      ( std::find(RO.localMap.begin(), RO.localMap.end(), gsfToEcal) == 
	RO.localMap.end() );
    bool haskf = 
      ( std::find(RO.localMap.begin(), RO.localMap.end(), kfToEcal) == 
	RO.localMap.end() );
    bool hasbrem = false;
    for( const auto& brem : RO.brems ) {
      ElementMap::value_type bremToEcal(brem.first,ecal.first);
      if( std::find(RO.localMap.begin(), RO.localMap.end(), bremToEcal) != 
	  RO.localMap.end() ) {
	hasbrem = true;
      }
    }
    if( hasbrem && ecal.first != RO.electronClusters[0] ) {      
      EcalBremEnergy += cenergy;
    } 
    if( !hasbrem && ecal.first != RO.electronClusters[0] ) {
      if( hasgsf ) OtherEcalGsfEnergy += cenergy;
      if( haskf  ) EcalBremEnergy += cenergy; // from conv. brem!
      if( !(hasgsf || haskf) ) OtherEcalGsfEnergy += cenergy; // stuff from SC
    }
  }
  
  if( RO.electronClusters[0] ) {
    reco::PFClusterRef cref = RO.electronClusters[0]->clusterRef();
    xtra.setGsfElectronClusterRef(_currentblock,*(RO.electronClusters[0]));
    FirstEcalGsfEnergy = cref->correctedEnergy();    
    deta_gsfecal = cref->positionREP().eta() - Etaout_gsf;
    gsfcluster.push_back(&*cref);
    PFClusterWidthAlgo pfwidth(gsfcluster);
    sigmaEtaEta = pfwidth.pflowSigmaEtaEta();
  } 

  // brem sequence information
  lateBrem = firstBrem = earlyBrem = -1.0f;
  if(RO.nBremsWithClusters > 0) {
    if (RO.lateBrem == 1) lateBrem = 1.0f;
    else lateBrem = 0.0f;
    firstBrem = RO.firstBrem;
    if(RO.firstBrem < 4) earlyBrem = 1.0f;
    else earlyBrem = 0.0f;
  }     
  xtra.setEarlyBrem(earlyBrem);
  xtra.setLateBrem(lateBrem);
  if( FirstEcalGsfEnergy > 0.0 ) {
    if( RefGSF.isNonnull() ) {
      xtra.setGsfTrackPout(gsfElement->Pout());
      // normalization observables
      const float Pt_gsf = RefGSF->ptMode();
      lnPt_gsf = std::log(Pt_gsf);
      Eta_gsf = RefGSF->etaMode();
      // tracking observables
      const double ptModeErrorGsf = RefGSF->ptModeError();
      dPtOverPt_gsf = (ptModeErrorGsf > 0. ? ptModeErrorGsf/Pt_gsf : 1.0);
      nhit_gsf = RefGSF->hitPattern().trackerLayersWithMeasurement();
      chi2_gsf = RefGSF->normalizedChi2();
      DPtOverPt_gsf =  (Pt_gsf - gsfElement->Pout().pt())/Pt_gsf;
      // kalman filter vars
      nhit_kf = 0;
      chi2_kf = -0.01;
      DPtOverPt_kf = -0.01;
      if( RefKF.isNonnull() ) {
	nhit_kf = RefKF->hitPattern().trackerLayersWithMeasurement();
	chi2_kf = RefKF->normalizedChi2();
	// not used for moment, weird behavior of variable
	// DPtOverPt_kf = (RefKF->pt() - RefKF->outerPt())/RefKF->pt();
      }	
      //tracker + calorimetry observables
      const double EcalETot = 
	(FirstEcalGsfEnergy+OtherEcalGsfEnergy+EcalBremEnergy);
      EtotPinMode  = EcalETot / Ein_gsf;
      EGsfPoutMode = FirstEcalGsfEnergy / Eout_gsf;
      EtotBremPinPoutMode = ( (EcalBremEnergy + OtherEcalGsfEnergy) / 
			      (Ein_gsf - Eout_gsf) );
      DEtaGsfEcalClust = std::abs(deta_gsfecal);
      SigmaEtaEta = std::log(sigmaEtaEta);
      xtra.setDeltaEta(DEtaGsfEcalClust);
      xtra.setSigmaEtaEta(sigmaEtaEta);      
      
      HOverHE = Ene_hcalgsf/(Ene_hcalgsf + FirstEcalGsfEnergy);
      HOverPin = Ene_hcalgsf / Ein_gsf;
      xtra.setHadEnergy(Ene_hcalgsf);

      // Apply bounds to variables and calculate MVA
      DPtOverPt_gsf = std::max(DPtOverPt_gsf,-0.2f);
      DPtOverPt_gsf =  std::min(DPtOverPt_gsf,1.0f);  
      dPtOverPt_gsf = std::min(dPtOverPt_gsf,0.3f);  
      chi2_gsf = std::min(chi2_gsf,10.0f);  
      DPtOverPt_kf = std::max(DPtOverPt_kf,-0.2f);
      DPtOverPt_kf = std::min(DPtOverPt_kf,1.0f);  
      chi2_kf = std::min(chi2_kf,10.0f);  
      EtotPinMode = std::max(EtotPinMode,0.0f);
      EtotPinMode = std::min(EtotPinMode,5.0f);  
      EGsfPoutMode = std::max(EGsfPoutMode,0.0f);
      EGsfPoutMode = std::min(EGsfPoutMode,5.0f);  
      EtotBremPinPoutMode = std::max(EtotBremPinPoutMode,0.0f);
      EtotBremPinPoutMode = std::min(EtotBremPinPoutMode,5.0f);  
      DEtaGsfEcalClust = std::min(DEtaGsfEcalClust,0.1f);  
      SigmaEtaEta = std::max(SigmaEtaEta,-14.0f);  
      HOverPin = std::max(HOverPin,0.0f);
      HOverPin = std::min(HOverPin,5.0f);
      /*
      std::cout << " **** PFEG BDT observables ****" << endl;
      std::cout << " < Normalization > " << endl;
      std::cout << " Pt_gsf " << Pt_gsf << " Pin " << Ein_gsf  
		<< " Pout " << Eout_gsf << " Eta_gsf " << Eta_gsf << endl;
      std::cout << " < PureTracking > " << endl;
      std::cout << " dPtOverPt_gsf " << dPtOverPt_gsf 
		<< " DPtOverPt_gsf " << DPtOverPt_gsf
		<< " chi2_gsf " << chi2_gsf
		<< " nhit_gsf " << nhit_gsf
		<< " DPtOverPt_kf " << DPtOverPt_kf
		<< " chi2_kf " << chi2_kf 
		<< " nhit_kf " << nhit_kf <<  endl;
      std::cout << " < track-ecal-hcal-ps " << endl;
      std::cout << " EtotPinMode " << EtotPinMode 
		<< " EGsfPoutMode " << EGsfPoutMode
		<< " EtotBremPinPoutMode " << EtotBremPinPoutMode
		<< " DEtaGsfEcalClust " << DEtaGsfEcalClust 
		<< " SigmaEtaEta " << SigmaEtaEta
		<< " HOverHE " << HOverHE << " Hcal energy " << Ene_hcalgsf
		<< " HOverPin " << HOverPin 
		<< " lateBrem " << lateBrem
		<< " firstBrem " << firstBrem << endl;
      */
      
      float vars[] = { lnPt_gsf, Eta_gsf, dPtOverPt_gsf, DPtOverPt_gsf, chi2_gsf,
                       nhit_kf, chi2_kf, EtotPinMode, EGsfPoutMode, EtotBremPinPoutMode,
                       DEtaGsfEcalClust, SigmaEtaEta, HOverHE, lateBrem, firstBrem };

      return hoc->gbrEle_->GetAdaBoostClassifier(vars);
    }
  }
  return -2.0f;
}

void PFEGammaAlgo::fill_extra_info( const ProtoEGObject& RO,
				    reco::PFCandidateEGammaExtra& xtra ) {
  // add tracks associated to clusters that are not T_FROM_GAMMACONV
  // info about single-leg convs is already save, so just veto in loops
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
    // go through non-conv-identified kfs and check MVA to add conversions
    for( auto kf = notconvkf; kf != notmatchedkf; ++kf ) {      
      const reco::PFBlockElementTrack* elemaskf =
	docast(const reco::PFBlockElementTrack*,kf->first);
      xtra.addExtraNonConvTrack(_currentblock,*elemaskf);   
    }
  }
}

// currently stolen from PFECALSuperClusterAlgo, we should
// try to factor this correctly since the operation is the same in
// both places...
reco::SuperCluster PFEGammaAlgo::
buildRefinedSuperCluster(const PFEGammaAlgo::ProtoEGObject& RO) {
  if( RO.ecalclusters.empty() ) { 
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
    PS1_clus_sum(0), PS2_clus_sum(0),
    ePS1(0), ePS2(0), ps1_energy(0.0), ps2_energy(0.0); 
  int condP1(1), condP2(1);
  for( auto& clus : RO.ecalclusters ) {
    ePS1 = 0;
    ePS2 = 0;
    isEE = PFLayer::ECAL_ENDCAP == clus.first->clusterRef()->layer();
    clusptr = 
      edm::refToPtr<reco::PFClusterCollection>(clus.first->clusterRef());
    bare_ptrs.push_back(clusptr.get());    

    const double cluseraw = clusptr->energy();
    double cluscalibe = clusptr->correctedEnergy();
    const math::XYZPoint& cluspos = clusptr->position();
    posX += cluseraw * cluspos.X();
    posY += cluseraw * cluspos.Y();
    posZ += cluseraw * cluspos.Z();
    // update EE calibrated super cluster energies
    if( isEE && RO.ecal2ps.count(clus.first)) {
      ePS1 = 0;
      ePS2 = 0;
      condP1 = condP2 = 1;

      const auto& psclusters = RO.ecal2ps.at(clus.first);
      
      for( auto i_ps = psclusters.begin(); i_ps != psclusters.end(); ++i_ps) {
	const PFClusterRef&  psclus = i_ps->first->clusterRef();
	
	auto const& recH_Frac = psclus->recHitFractions();	
	
	switch( psclus->layer() ) {
	case PFLayer::PS1:
	  for (auto const& recH : recH_Frac){
	    ESDetId strip1 = recH.recHitRef()->detId();
	    if(strip1 != ESDetId(0)){
	      ESChannelStatusMap::const_iterator status_p1 = channelStatus_->getMap().find(strip1);
	      //getStatusCode() == 0 => active channel
	      // apply correction if all recHits are dead
	      if(status_p1->getStatusCode() == 0) condP1 = 0;
	    }
	  }
	  break;
	case PFLayer::PS2:
	  for (auto const& recH : recH_Frac){
	    ESDetId strip2 = recH.recHitRef()->detId();
	    if(strip2 != ESDetId(0)) {
	      ESChannelStatusMap::const_iterator status_p2 = channelStatus_->getMap().find(strip2);
	      if(status_p2->getStatusCode() == 0) condP2 = 0;
	    }
	  }
	  break;
	default:
	  break;
	}
      }
      
      
      PS1_clus_sum = std::accumulate(psclusters.begin(),psclusters.end(),
				     0.0,sumps1);
      PS2_clus_sum = std::accumulate(psclusters.begin(),psclusters.end(),
				     0.0,sumps2);
            
      if(condP1 == 1) ePS1 = -1.;
      if(condP2 == 1) ePS2 = -1.;

      cluscalibe = 
	cfg_.thePFEnergyCalibration->energyEm(*clusptr,
					      PS1_clus_sum,PS2_clus_sum,
					      ePS1, ePS2,
					      cfg_.applyCrackCorrections);
    }
    if(ePS1 == -1.) ePS1 = 0;
    if(ePS2 == -1.) ePS2 = 0;

    rawSCEnergy  += cluseraw;
    corrSCEnergy += cluscalibe;    
    ps1_energy   += ePS1;
    ps2_energy   += ePS2;
    corrPSEnergy += ePS1 + ePS2;    
  }
  posX /= rawSCEnergy;
  posY /= rawSCEnergy;
  posZ /= rawSCEnergy;

  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy,math::XYZPoint(posX,posY,posZ)); 

  clusptr = 
    edm::refToPtr<reco::PFClusterCollection>(RO.ecalclusters.front().
					     first->clusterRef());
  new_sc.setCorrectedEnergy(corrSCEnergy);
  new_sc.setSeed(clusptr);
  new_sc.setPreshowerEnergyPlane1(ps1_energy);
  new_sc.setPreshowerEnergyPlane2(ps2_energy);
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
    if( RO.ecal2ps.count(clus.first) ) {
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
	  new_sc.addPreshowerCluster(psclus);
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
  }
  
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
  if( RO.primaryGSFs.empty() ) return;  
  // need energy sums to tell if we've added crap or not
  const double Pin_gsf = RO.primaryGSFs.front().first->GsftrackRef()->pMode();
  const double gsfOuterEta  = 
    RO.primaryGSFs.front().first->positionAtECALEntrance().Eta();
  double tot_ecal= 0.0;  
  std::vector<double> min_brem_dists;
  std::vector<double> closest_brem_eta;    
  // first get the total ecal energy (we should replace this with a cache)
  for( const auto& ecal : RO.ecalclusters ) {
    tot_ecal += ecal.first->clusterRef()->correctedEnergy();
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
  for( auto secd_kf = RO.secondaryKFs.begin(); 
       secd_kf != RO.secondaryKFs.end(); ++secd_kf ) {
    reco::TrackRef trkRef =   secd_kf->first->trackRef();   
    const float secpin = secd_kf->first->trackRef()->p();  
    bool remove_this_kf = false;
    for( auto ecal = RO.ecalclusters.begin(); 
	 ecal != RO.ecalclusters.end(); ++ecal ) {
      size_t bremidx = std::distance(RO.ecalclusters.begin(),ecal);
      const float minbremdist = min_brem_dists[bremidx];
      const double ecalenergy = ecal->first->clusterRef()->correctedEnergy();
      const double Epin = ecalenergy/secpin;
      const double detaGsf = 
	std::abs(gsfOuterEta - ecal->first->clusterRef()->positionREP().Eta());
      const double detaBrem = 
	std::abs(closest_brem_eta[bremidx] - 
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
	    remove_this_kf = true;   
	    ecal = RO.ecalclusters.erase(ecal);
	    if( ecal == RO.ecalclusters.end() ) break;	    
	}
      }
    }
    if( remove_this_kf ) {
      secd_kf = RO.secondaryKFs.erase(secd_kf);
      if( secd_kf == RO.secondaryKFs.end() ) break;
    }
  }  
}

void PFEGammaAlgo::
unlinkRefinableObjectKFandECALMatchedToHCAL(ProtoEGObject& RO,
					    bool removeFreeECAL,
					    bool removeSCEcal) {
  std::vector<bool> cluster_in_sc;    
  auto ecal_begin = RO.ecalclusters.begin();
  auto ecal_end   = RO.ecalclusters.end();
  auto hcal_begin = _splayedblock[reco::PFBlockElement::HCAL].begin();
  auto hcal_end   = _splayedblock[reco::PFBlockElement::HCAL].end();
  for( auto secd_kf = RO.secondaryKFs.begin(); 
       secd_kf != RO.secondaryKFs.end(); ++secd_kf ) {
    bool remove_this_kf = false;
    NotCloserToOther<reco::PFBlockElement::TRACK,reco::PFBlockElement::HCAL>
      tracksToHCALs(_currentblock,_currentlinks,secd_kf->first);
    reco::TrackRef trkRef =   secd_kf->first->trackRef();

    bool goodTrack = PFTrackAlgoTools::isGoodForEGM(trkRef->algo());
    const float secpin = trkRef->p();       
    
    for( auto ecal = ecal_begin; ecal != ecal_end; ++ecal ) {
      const double ecalenergy = ecal->first->clusterRef()->correctedEnergy();
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
	  const bool isHoHE = ( (hcalenergy / hpluse ) > 0.1 && goodTrack );
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
		<< " Algo Track " << trkRef->algo() << std::endl;
	      remove_this_kf = true;
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
		<< " Algo Track " <<trkRef->algo() << std::endl;
	      remove_this_kf = true;
	    }
	  }  
	}
      }
    }
    if( remove_this_kf ) {
      secd_kf = RO.secondaryKFs.erase(secd_kf);
      if( secd_kf == RO.secondaryKFs.end() ) break;
    }
  }  
}



bool PFEGammaAlgo::isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
				    const reco::PFBlockElementGsfTrack& GsfEl) {
  bool isPrimary = false;
  
  const GsfPFRecTrackRef& gsfPfRef = GsfEl.GsftrackRefPF();
  
  if(gsfPfRef.isNonnull()) {
    const PFRecTrackRef&  kfPfRef = KfEl.trackRefPF();
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

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrackAndHFLinker : public BlockElementLinkerBase {
public:
  TrackAndHFLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  double testLink 
  ( const reco::PFBlockElement*,
    const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndHFLinker, 
		  "TrackAndHFLinker");

double TrackAndHFLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  

  //KH: should consider extrapolating to the face of HF short fibers for HF HAD???
  constexpr reco::PFTrajectoryPoint::LayerType VFcalEntrance =
    reco::PFTrajectoryPoint::VFcalEntrance;
  const reco::PFBlockElementCluster *hfelem(nullptr);
  const reco::PFBlockElementTrack   *tkelem(nullptr);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    hfelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    hfelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = hfelem->clusterRef();
  const reco::PFCluster::REPPoint& hfreppos = clusterref->positionREP(); 

  const reco::PFTrajectoryPoint& tkAtHF =
    trackref->extrapolatedPoint( VFcalEntrance );
  if ( _useKDTree && hfelem->isMultilinksValide() ) { //KDTree Algo
    const reco::PFMultilinksType& multilinks = hfelem->getMultilinks();
    const double tracketa = tkAtHF.positionREP().Eta();
    const double trackphi = tkAtHF.positionREP().Phi();
    
    // Check if the link Track/HF exist
    reco::PFMultilinksType::const_iterator mlit = multilinks.begin();
    for (; mlit != multilinks.end(); ++mlit)
      if ((mlit->first == trackphi) && (mlit->second == tracketa))
	break;
    
    // If the link exist, we fill dist and linktest.     

    if (mlit != multilinks.end()){     

      edm::LogWarning("TrackHFLinker ") <<"Special case of linking with track and HF clusters and found multiple links ";
      //std::cout << tracketa << " " << trackphi << std::endl;
      //std::cout << clusterref->eta() << " " << clusterref->phi() << std::endl;

      dist = LinkByRecHit::computeDist(hfreppos.Eta(), 
				       hfreppos.Phi(), 
				       tracketa, 
				       trackphi);

      
    } else {
    if ( tkAtHF.isValid() )
      dist = LinkByRecHit::testTrackAndClusterByRecHit( *trackref, 
							*clusterref,
							false, _debug );
    }
  }
  return dist;
}


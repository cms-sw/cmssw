#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndHCALLinker : public BlockElementLinkerBase {
public:
  GSFAndHCALLinker(const edm::ParameterSet& conf) :
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
		  GSFAndHCALLinker, 
		  "GSFAndHCALLinker");

double GSFAndHCALLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  constexpr reco::PFTrajectoryPoint::LayerType HCALEnt =
    reco::PFTrajectoryPoint::HCALEntrance;
  const reco::PFBlockElementCluster  *hcalelem(nullptr);
  const reco::PFBlockElementGsfTrack *gsfelem(nullptr);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    gsfelem  = static_cast<const reco::PFBlockElementGsfTrack*>(elem2);
  } else {
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    gsfelem  = static_cast<const reco::PFBlockElementGsfTrack*>(elem1);
  }
  const reco::PFRecTrack& track = gsfelem->GsftrackPF();
  const reco::PFClusterRef& clusterref = hcalelem->clusterRef();
  const reco::PFTrajectoryPoint& tkAtHCAL =
    track.extrapolatedPoint( HCALEnt );
  if( tkAtHCAL.isValid() ) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit( track, *clusterref, 
						      false, _debug );
  }
  if ( _debug ) {
    if ( dist > 0. ) {
      std::cout << " Here a link has been established" 
		<< " between a GSF track an Hcal with dist  " 
		<< dist <<  std::endl;
    } else {
      if( _debug ) std::cout << " No link found " << std::endl;
    }
  }
  
  return dist;
}

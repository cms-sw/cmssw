#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/Math/interface/deltaPhi.h"

class TrackAndHCALLinker : public BlockElementLinkerBase {
public:
  TrackAndHCALLinker(const edm::ParameterSet& conf) :
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
		  TrackAndHCALLinker, 
		  "TrackAndHCALLinker");

double TrackAndHCALLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  constexpr reco::PFTrajectoryPoint::LayerType HCALEntrance =
    reco::PFTrajectoryPoint::HCALEntrance;
  constexpr reco::PFTrajectoryPoint::LayerType HCALExit =
    reco::PFTrajectoryPoint::HCALExit;
  const reco::PFBlockElementCluster *hcalelem(NULL);
  const reco::PFBlockElementTrack   *tkelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = hcalelem->clusterRef();
  const reco::PFCluster::REPPoint& hcalreppos = clusterref->positionREP(); 
  const reco::PFTrajectoryPoint& tkAtHCALEnt =
    trackref->extrapolatedPoint( HCALEntrance );
  const reco::PFTrajectoryPoint& tkAtHCALEx =
    trackref->extrapolatedPoint( HCALExit );
  const double dHEta = ( tkAtHCALEx.positionREP().Eta() - 
			 tkAtHCALEnt.positionREP().Eta()  );
  const double dHPhi = reco::deltaPhi( tkAtHCALEx.positionREP().Phi(), 
				       tkAtHCALEnt.positionREP().Phi() );
  if ( _useKDTree && hcalelem->isMultilinksValide() ) { //KDTree Algo
    const reco::PFMultilinksType& multilinks = hcalelem->getMultilinks();
    const double tracketa = tkAtHCALEnt.positionREP().Eta();
    const double trackphi = tkAtHCALEnt.positionREP().Phi();
    
    // Check if the link Track/Hcal exist
    reco::PFMultilinksType::const_iterator mlit = multilinks.begin();
    for (; mlit != multilinks.end(); ++mlit)
      if ((mlit->first == trackphi) && (mlit->second == tracketa))
	break;
    
    // If the link exist, we fill dist and linktest.     
    if (mlit != multilinks.end()){     
      if (tkAtHCALEx.position().R()<tkAtHCALEnt.position().R()) {
	throw cms::Exception("WorkingIn4DSpace")
	  << "Qu'est ce que c'est que ce gag ? " 
	  << tkAtHCALEx.position().R() << " is smaller than " 
	  << tkAtHCALEnt.position().R() << " !" << std::endl;
      }      
      dist = LinkByRecHit::computeDist(hcalreppos.Eta(), 
				       hcalreppos.Phi(), 
				       tracketa + 0.1 * dHEta, 
				       trackphi + 0.1 * dHPhi);
    }
  } else {// Old algorithm
    if ( tkAtHCALEnt.isValid() )
      dist = LinkByRecHit::testTrackAndClusterByRecHit( *trackref, 
							*clusterref,
							false, _debug );
  }
  return dist;
}

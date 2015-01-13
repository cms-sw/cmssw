#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class TrackAndHGCHEBLinker : public BlockElementLinkerBase {
public:
  TrackAndHGCHEBLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  bool linkPrefilter( const reco::PFBlockElement*,
		      const reco::PFBlockElement* ) const override;

  double testLink( const reco::PFBlockElement*,
		   const reco::PFBlockElement* ) const override;

private:
  const bool _useKDTree,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndHGCHEBLinker, 
		  "TrackAndHGCHEBLinker");

bool TrackAndHGCHEBLinker::
linkPrefilter( const reco::PFBlockElement* elem1,
	       const reco::PFBlockElement* elem2 ) const {    
  bool result = false;
  switch( elem1->type() ){ 
  case reco::PFBlockElement::TRACK:
    result = (elem1->isMultilinksValide() && elem1->getMultilinks().size() > 0);
    break;
  case reco::PFBlockElement::HGC_HCALB:
    result = (elem2->isMultilinksValide() && elem2->getMultilinks().size() > 0);
  default:
    break;
  } 
  return (_useKDTree ? result : true);  
}

double TrackAndHGCHEBLinker::
testLink( const reco::PFBlockElement* elem1,
	  const reco::PFBlockElement* elem2 ) const {  
  constexpr reco::PFTrajectoryPoint::LayerType HGCEntrance =
    reco::PFTrajectoryPoint::HGC_ECALEntrance;
  const reco::PFBlockElementCluster *hgchebelem(NULL);
  const reco::PFBlockElementTrack   *tkelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    hgchebelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    hgchebelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = hgchebelem->clusterRef();
  const reco::PFCluster::REPPoint& clusreppos = clusterref->positionREP(); 
  const reco::PFTrajectoryPoint& tkAtHGCHEB =
    trackref->extrapolatedPoint( HGCEntrance );
   const reco::PFCluster::REPPoint& tkreppos = tkAtHGCHEB.positionREP();

  // Check if the linking has been done using the KDTree algo
  // Glowinski & Gouzevitch
  if ( _useKDTree && tkelem->isMultilinksValide() ) { //KDTree Algo    
    const reco::PFMultilinksType& multilinks = tkelem->getMultilinks();
    const double ecalphi = clusreppos.Phi();
    const double ecaleta = clusreppos.Eta();
    
    // Check if the link Track/Ecal exist
    reco::PFMultilinksType::const_iterator mlit = multilinks.begin();
    for (; mlit != multilinks.end(); ++mlit)
      if ((mlit->first == ecalphi) && (mlit->second == ecaleta))
	break;
    
    // If the link exist, we fill dist and linktest. 
    if (mlit != multilinks.end()){      
      dist = LinkByRecHit::computeDist(ecaleta, ecalphi, 
				       tkreppos.Eta(), tkreppos.Phi());
    }    
  }
  return dist;
}

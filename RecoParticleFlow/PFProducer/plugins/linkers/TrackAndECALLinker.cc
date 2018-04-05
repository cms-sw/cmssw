#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class TrackAndECALLinker : public BlockElementLinkerBase {
public:
  TrackAndECALLinker(const edm::ParameterSet& conf) :
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
		  TrackAndECALLinker, 
		  "TrackAndECALLinker");

bool TrackAndECALLinker::
linkPrefilter( const reco::PFBlockElement* elem1,
	       const reco::PFBlockElement* elem2 ) const {  
  bool result = false;
  switch( elem1->type() ){ 
  case reco::PFBlockElement::TRACK:
    result = (elem1->isMultilinksValide() && !elem1->getMultilinks().empty());
    break;
  case reco::PFBlockElement::ECAL:
    result = (elem2->isMultilinksValide() && !elem2->getMultilinks().empty());
  default:
    break;
  } 
  return (_useKDTree ? result : true);  
}

double TrackAndECALLinker::
testLink( const reco::PFBlockElement* elem1,
	  const reco::PFBlockElement* elem2 ) const {  
  constexpr reco::PFTrajectoryPoint::LayerType ECALShowerMax =
    reco::PFTrajectoryPoint::ECALShowerMax;
  const reco::PFBlockElementCluster *ecalelem(nullptr);
  const reco::PFBlockElementTrack   *tkelem(nullptr);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = ecalelem->clusterRef();
  const reco::PFCluster::REPPoint& ecalreppos = clusterref->positionREP(); 
  const reco::PFTrajectoryPoint& tkAtECAL =
    trackref->extrapolatedPoint( ECALShowerMax );
   const reco::PFCluster::REPPoint& tkreppos = tkAtECAL.positionREP();

  // Check if the linking has been done using the KDTree algo
  // Glowinski & Gouzevitch
  if ( _useKDTree && tkelem->isMultilinksValide() ) { //KDTree Algo    
    const reco::PFMultilinksType& multilinks = tkelem->getMultilinks();
    const double ecalphi = ecalreppos.Phi();
    const double ecaleta = ecalreppos.Eta();
    
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
    
  } else {// Old algorithm
    if ( tkAtECAL.isValid() )
      dist = LinkByRecHit::testTrackAndClusterByRecHit( *trackref, 
							*clusterref, 
							false, _debug );
  }
  
  if ( _debug ) { 
    if( dist > 0. ) { 
      std::cout << " Here a link has been established"
		<< " between a track an Ecal with dist  " 
		<< dist <<  std::endl;
    } else
      std::cout << " No link found " << std::endl;
  }  
  return dist;
}

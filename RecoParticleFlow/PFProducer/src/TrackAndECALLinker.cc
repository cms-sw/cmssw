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
  
  double operator() 
  ( const std::unique_ptr<reco::PFBlockElement>&,
    const std::unique_ptr<reco::PFBlockElement>& ) const override;

private:
  bool _useKDTree,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndECALLinker, 
		  "TrackAndECALLinker");

double TrackAndECALLinker::operator()
  ( const std::unique_ptr<reco::PFBlockElement>& elem1,
    const std::unique_ptr<reco::PFBlockElement>& elem2) const {  
  constexpr reco::PFTrajectoryPoint::LayerType ECALShowerMax =
    reco::PFTrajectoryPoint::ECALShowerMax;
  const reco::PFBlockElementCluster *ecalelem(NULL);
  const reco::PFBlockElementTrack   *tkelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1.get());
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2.get());
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2.get());
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1.get());
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
    else
      dist = -1.;
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

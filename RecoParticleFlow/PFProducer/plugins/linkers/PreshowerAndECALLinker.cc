#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class PreshowerAndECALLinker : public BlockElementLinkerBase {
public:
  PreshowerAndECALLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  bool linkPrefilter( const reco::PFBlockElement*,
		      const reco::PFBlockElement* ) const override;

  double testLink( const reco::PFBlockElement*,
		   const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  PreshowerAndECALLinker, 
		  "PreshowerAndECALLinker");

bool PreshowerAndECALLinker::
linkPrefilter( const reco::PFBlockElement* elem1,
	       const reco::PFBlockElement* elem2 ) const {  
  bool result = false;
  switch( elem1->type() ){
  case reco::PFBlockElement::PS1:
  case reco::PFBlockElement::PS2:
    result = ( elem1->isMultilinksValide() && 
	       elem1->getMultilinks().size() > 0 );    
    break;
  case reco::PFBlockElement::ECAL:
    result = ( elem2->isMultilinksValide() && 
	       elem2->getMultilinks().size() > 0 );
    break;
  default:
    break;
  }   
  return (_useKDTree ? result : true);
}

double PreshowerAndECALLinker::
testLink( const reco::PFBlockElement* elem1,
	  const reco::PFBlockElement* elem2) const {  
  const reco::PFBlockElementCluster *pselem(NULL), *ecalelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    pselem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    pselem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& psref = pselem->clusterRef();
  const reco::PFClusterRef& ecalref = ecalelem->clusterRef();  
  if( psref.isNull() || ecalref.isNull() ) {
    throw cms::Exception("BadClusterRefs") 
      << "PFBlockElementCluster's refs are null!";
  }  
  // Check if the linking has been done using the KDTree algo
  // Glowinski & Gouzevitch
  if ( _useKDTree && pselem->isMultilinksValide() ) { // KDTree algo
    const reco::PFMultilinksType& multilinks = pselem->getMultilinks();	
    const reco::PFCluster::REPPoint&  ecalreppos = ecalref->positionREP();
    const math::XYZPoint& ecalxyzpos = ecalref->position();
    const math::XYZPoint& psxyzpos = psref->position();
    const double ecalPhi = ecalreppos.Phi();
    const double ecalEta = ecalreppos.Eta();
    
    // Check if the link PS/Ecal exist
    reco::PFMultilinksType::const_iterator mlit = multilinks.begin();
    for (; mlit != multilinks.end(); ++mlit)
      if ((mlit->first == ecalPhi) && (mlit->second == ecalEta))
	break;
    
    // If the link exist, we fill dist and linktest. 
    if (mlit != multilinks.end()){      
      dist = 
	LinkByRecHit::computeDist(ecalxyzpos.X()/1000.,ecalxyzpos.Y()/1000.,
				  psxyzpos.X()/1000.  ,psxyzpos.Y()/1000., 
				  false);
    }   
  } else { //Old algorithm
    dist = LinkByRecHit::testECALAndPSByRecHit( *ecalref, *psref ,_debug);
  }
  return dist;
}

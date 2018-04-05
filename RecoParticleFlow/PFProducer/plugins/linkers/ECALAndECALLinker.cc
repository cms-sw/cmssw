#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class ECALAndECALLinker : public BlockElementLinkerBase {
public:
  ECALAndECALLinker(const edm::ParameterSet& conf) :
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
		  ECALAndECALLinker, 
		  "ECALAndECALLinker");

bool ECALAndECALLinker::
linkPrefilter( const reco::PFBlockElement* elem1,
	       const reco::PFBlockElement* elem2) const { 
  const reco::PFBlockElementCluster* ecal1 = 
    static_cast<const reco::PFBlockElementCluster*>(elem1);
  const reco::PFBlockElementCluster* ecal2 = 
    static_cast<const reco::PFBlockElementCluster*>(elem2);
  return ( ecal1->superClusterRef().isNonnull() && 
	   ecal2->superClusterRef().isNonnull()    ); 
}

double ECALAndECALLinker::
testLink( const reco::PFBlockElement* elem1,
	  const reco::PFBlockElement* elem2) const { 
  double dist = -1.0;
  
  const reco::PFBlockElementCluster* ecal1 = 
    static_cast<const reco::PFBlockElementCluster*>(elem1);
  const reco::PFBlockElementCluster* ecal2 = 
    static_cast<const reco::PFBlockElementCluster*>(elem2);
 
  const reco::SuperClusterRef& sc1 = ecal1->superClusterRef();
  const reco::SuperClusterRef& sc2 = ecal2->superClusterRef();

  const reco::PFClusterRef& clus1 = ecal1->clusterRef();
  const reco::PFClusterRef& clus2 = ecal2->clusterRef();

  if( sc1.isNonnull() && sc2.isNonnull() && sc1 == sc2 ) {    
    dist=LinkByRecHit::computeDist( clus1->positionREP().Eta(),
				    clus1->positionREP().Phi(), 
				    clus2->positionREP().Eta(), 
				    clus2->positionREP().Phi() );
  }

  return dist;
}

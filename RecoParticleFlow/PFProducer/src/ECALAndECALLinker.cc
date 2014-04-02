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
  
  double operator() 
  ( const std::unique_ptr<reco::PFBlockElement>&,
    const std::unique_ptr<reco::PFBlockElement>& ) const override;

private:
  bool _useKDTree,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  ECALAndECALLinker, 
		  "ECALAndECALLinker");

double ECALAndECALLinker::operator()
  ( const std::unique_ptr<reco::PFBlockElement>& elem1,
    const std::unique_ptr<reco::PFBlockElement>& elem2) const { 
  double dist = -1.0;
  
  const reco::PFBlockElementCluster* ecal1 = 
    static_cast<const reco::PFBlockElementCluster*>(elem1.get());
  const reco::PFBlockElementCluster* ecal2 = 
    static_cast<const reco::PFBlockElementCluster*>(elem2.get());
 
  const reco::PFClusterRef& clus1 = ecal1->clusterRef();
  const reco::PFClusterRef& clus2 = ecal2->clusterRef();

  if( ecal1->superClusterRef() == ecal2->superClusterRef()) {
    dist=LinkByRecHit::computeDist( clus1->positionREP().Eta(),
				    clus1->positionREP().Phi(), 
				    clus2->positionREP().Eta(), 
				    clus2->positionREP().Phi() );
  }

  return dist;
}

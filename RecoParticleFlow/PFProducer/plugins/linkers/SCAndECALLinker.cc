#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

class SCAndECALLinker : public BlockElementLinkerBase {
public:
  SCAndECALLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)),
        superClusterMatchByRef_(conf.getParameter<bool>("SuperClusterMatchByRef")) {}

  double testLink(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

private:
  bool useKDTree_, debug_, superClusterMatchByRef_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, SCAndECALLinker, "SCAndECALLinker");

double SCAndECALLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  double dist = -1.0;
  const reco::PFBlockElementCluster* ecalelem(nullptr);
  const reco::PFBlockElementSuperCluster* scelem(nullptr);
  if (elem1->type() < elem2->type()) {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    scelem = static_cast<const reco::PFBlockElementSuperCluster*>(elem2);
  } else {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    scelem = static_cast<const reco::PFBlockElementSuperCluster*>(elem1);
  }
  const reco::PFClusterRef& clus = ecalelem->clusterRef();
  const reco::SuperClusterRef& sclus = scelem->superClusterRef();
  if (sclus.isNull()) {
    throw cms::Exception("BadRef") << "SuperClusterRef is invalid!";
  }

  if (superClusterMatchByRef_) {
    if (sclus == ecalelem->superClusterRef())
      dist = 0.001;
  } else {
    if (ClusterClusterMapping::overlap(*sclus, *clus)) {
      dist = LinkByRecHit::computeDist(
          sclus->position().eta(), sclus->position().phi(), clus->positionREP().Eta(), clus->positionREP().Phi());
    }
  }
  return dist;
}

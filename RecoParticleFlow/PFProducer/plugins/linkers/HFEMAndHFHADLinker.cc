#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class HFEMAndHFHADLinker : public BlockElementLinkerBase {
public:
  HFEMAndHFHADLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  double testLink(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type type1,
                  reco::PFBlockElement::Type type2,
                  const ElementListConst& elements,
                  const PFTables& tables,
                  const reco::PFMultiLinksIndex& multilinks) const override;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, HFEMAndHFHADLinker, "HFEMAndHFHADLinker");

double HFEMAndHFHADLinker::testLink(size_t ielem1,
                                    size_t ielem2,
                                    reco::PFBlockElement::Type type1,
                                    reco::PFBlockElement::Type type2,
                                    const ElementListConst& elements,
                                    const PFTables& tables,
                                    const reco::PFMultiLinksIndex& multilinks) const {
  const auto* elem1 = elements[ielem1];
  const auto* elem2 = elements[ielem2];
  const reco::PFBlockElementCluster *hfemelem(nullptr), *hfhadelem(nullptr);
  if (type1 < type2) {
    hfemelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    hfhadelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    hfemelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    hfhadelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& hfemref = hfemelem->clusterRef();
  const reco::PFClusterRef& hfhadref = hfhadelem->clusterRef();
  if (hfemref.isNull() || hfhadref.isNull()) {
    throw cms::Exception("BadClusterRefs") << "PFBlockElementCluster's refs are null!";
  }
  return LinkByRecHit::testHFEMAndHFHADByRecHit(*hfemref, *hfhadref, debug_);
}

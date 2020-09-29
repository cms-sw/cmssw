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
  size_t ihfem_elem = 0;
  size_t ihfhad_elem = 0;

  if (type1 < type2) {
    ihfem_elem = ielem1;
    ihfhad_elem = ielem2;
  } else {
    ihfem_elem = ielem2;
    ihfhad_elem = ielem1;
  }
  const size_t ihfem = tables.clusters_hfem.element_to_cluster[ihfem_elem];
  const size_t ihfhad = tables.clusters_hfhad.element_to_cluster[ihfhad_elem];

  return LinkByRecHit::testHFEMAndHFHADByRecHit(
      ihfem, ihfhad, tables.clusters_hfem.cluster_table, tables.clusters_hfhad.cluster_table);
}

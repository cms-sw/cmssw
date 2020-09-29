#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class HCALAndHOLinker : public BlockElementLinkerBase {
public:
  HCALAndHOLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, HCALAndHOLinker, "HCALAndHOLinker");

double HCALAndHOLinker::testLink(size_t ielem1,
                                 size_t ielem2,
                                 reco::PFBlockElement::Type type1,
                                 reco::PFBlockElement::Type type2,
                                 const ElementListConst& elements,
                                 const PFTables& tables,
                                 const reco::PFMultiLinksIndex& multilinks) const {
  using namespace edm::soa::col;
  double dist(-1.0);

  size_t ihcal_elem = 0;
  size_t iho_elem = 0;

  if (type1 < type2) {
    ihcal_elem = ielem1;
    iho_elem = ielem2;
  } else {
    ihcal_elem = ielem2;
    iho_elem = ielem1;
  }
  const size_t ihcal = tables.clusters_hcal.element_to_cluster[ihcal_elem];
  const size_t iho = tables.clusters_ho.element_to_cluster[iho_elem];

  const auto hcal_eta = tables.clusters_hcal.cluster_table.get<pf::cluster::Eta>(ihcal);
  const auto hcal_phi = tables.clusters_hcal.cluster_table.get<pf::cluster::Phi>(ihcal);
  const auto ho_eta = tables.clusters_ho.cluster_table.get<pf::cluster::Eta>(iho);
  const auto ho_phi = tables.clusters_ho.cluster_table.get<pf::cluster::Phi>(iho);

  dist = (std::abs(hcal_eta) < 1.5 ? LinkByRecHit::computeDist(hcal_eta, hcal_phi, ho_eta, ho_phi) : -1.0);
  return (dist < 0.2 ? dist : -1.0);
}

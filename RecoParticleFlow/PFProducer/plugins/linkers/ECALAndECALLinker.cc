#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class ECALAndECALLinker : public BlockElementLinkerBase {
public:
  ECALAndECALLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  bool linkPrefilter(size_t ielem1,
                     size_t ielem2,
                     reco::PFBlockElement::Type type1,
                     reco::PFBlockElement::Type type2,
                     const reco::PFMultiLinksIndex& multilinks,
                     const reco::PFBlockElement*,
                     const reco::PFBlockElement*) const override;

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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, ECALAndECALLinker, "ECALAndECALLinker");

bool ECALAndECALLinker::linkPrefilter(size_t ielem1,
                                      size_t ielem2,
                                      reco::PFBlockElement::Type type1,
                                      reco::PFBlockElement::Type type2,
                                      const reco::PFMultiLinksIndex& multilinks,
                                      const reco::PFBlockElement* elem1,
                                      const reco::PFBlockElement* elem2) const {
  const reco::PFBlockElementCluster* ecal1 = static_cast<const reco::PFBlockElementCluster*>(elem1);
  const reco::PFBlockElementCluster* ecal2 = static_cast<const reco::PFBlockElementCluster*>(elem2);
  return (ecal1->superClusterRef().isNonnull() && ecal2->superClusterRef().isNonnull());
}

double ECALAndECALLinker::testLink(size_t ielem1,
                                   size_t ielem2,
                                   reco::PFBlockElement::Type type1,
                                   reco::PFBlockElement::Type type2,
                                   const ElementListConst& elements,
                                   const PFTables& tables,
                                   const reco::PFMultiLinksIndex& multilinks) const {
  using namespace edm::soa::col;

  const size_t ielem1_ecal = tables.clusters_ecal_.element_to_cluster_[ielem1];
  const size_t ielem2_ecal = tables.clusters_ecal_.element_to_cluster_[ielem2];

  const auto& ecal_table = tables.clusters_ecal_.cluster_table_;

  double dist = -1.0;

  if (ecal_table.get<pf::cluster::SCRefKey>(ielem1_ecal) == ecal_table.get<pf::cluster::SCRefKey>(ielem2_ecal)) {
    dist = LinkByRecHit::computeDist(ecal_table.get<pf::cluster::Eta>(ielem1_ecal),
                                     ecal_table.get<pf::cluster::Phi>(ielem1_ecal),
                                     ecal_table.get<pf::cluster::Eta>(ielem2_ecal),
                                     ecal_table.get<pf::cluster::Phi>(ielem2_ecal));
  }

  return dist;
}

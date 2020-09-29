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
                     const PFTables& tables,
                     const reco::PFMultiLinksIndex& multilinks) const override;

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
                                      const PFTables& tables,
                                      const reco::PFMultiLinksIndex& multilinks) const {
  using SCRefIsNonNull = edm::soa::col::pf::cluster::SCRefIsNonNull;

  const auto& cl_ecal = tables.clusters_ecal;
  const auto& ecal_table = cl_ecal.cluster_table;
  const size_t ielem1_ecal = cl_ecal.element_to_cluster[ielem1];
  const size_t ielem2_ecal = cl_ecal.element_to_cluster[ielem2];

  return (ecal_table.get<SCRefIsNonNull>(ielem1_ecal) && ecal_table.get<SCRefIsNonNull>(ielem2_ecal));
}

double ECALAndECALLinker::testLink(size_t ielem1,
                                   size_t ielem2,
                                   reco::PFBlockElement::Type type1,
                                   reco::PFBlockElement::Type type2,
                                   const ElementListConst& elements,
                                   const PFTables& tables,
                                   const reco::PFMultiLinksIndex& multilinks) const {
  using Eta = edm::soa::col::pf::cluster::Eta;
  using Phi = edm::soa::col::pf::cluster::Phi;
  using SCRefKey = edm::soa::col::pf::cluster::SCRefKey;

  const auto& cl_ecal = tables.clusters_ecal;
  const auto& ecal_table = cl_ecal.cluster_table;

  const size_t ielem1_ecal = cl_ecal.element_to_cluster[ielem1];
  const size_t ielem2_ecal = cl_ecal.element_to_cluster[ielem2];

  double dist = -1.0;

  if (ecal_table.get<SCRefKey>(ielem1_ecal) == ecal_table.get<SCRefKey>(ielem2_ecal)) {
    dist = LinkByRecHit::computeDist(ecal_table.get<Eta>(ielem1_ecal),
                                     ecal_table.get<Phi>(ielem1_ecal),
                                     ecal_table.get<Eta>(ielem2_ecal),
                                     ecal_table.get<Phi>(ielem2_ecal));
  }

  return dist;
}

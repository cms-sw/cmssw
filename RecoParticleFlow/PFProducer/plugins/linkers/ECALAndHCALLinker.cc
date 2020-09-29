#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class ECALAndHCALLinker : public BlockElementLinkerBase {
public:
  ECALAndHCALLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        minAbsEtaEcal_(conf.getParameter<double>("minAbsEtaEcal")),
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
  double minAbsEtaEcal_;
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, ECALAndHCALLinker, "ECALAndHCALLinker");

double ECALAndHCALLinker::testLink(size_t ielem1,
                                   size_t ielem2,
                                   reco::PFBlockElement::Type type1,
                                   reco::PFBlockElement::Type type2,
                                   const ElementListConst& elements,
                                   const PFTables& tables,
                                   const reco::PFMultiLinksIndex& multilinks) const {
  using Eta = edm::soa::col::pf::cluster::Eta;
  using Phi = edm::soa::col::pf::cluster::Phi;
  double dist(-1.0);

  size_t iecal_elem = 0;
  size_t ihcal_elem = 0;

  if (type1 < type2) {
    iecal_elem = ielem1;
    ihcal_elem = ielem2;
  } else {
    iecal_elem = ielem2;
    ihcal_elem = ielem1;
  }
  const auto& ch = tables.clusters_hcal;
  const auto& ce = tables.clusters_ecal;
  const size_t iecal = ce.element_to_cluster[iecal_elem];
  const size_t ihcal = ch.element_to_cluster[ihcal_elem];

  dist = (std::abs(ce.cluster_table.get<Eta>(iecal)) > minAbsEtaEcal_
              ? LinkByRecHit::computeDist(ce.cluster_table.get<Eta>(iecal),
                                          ce.cluster_table.get<Phi>(iecal),
                                          ch.cluster_table.get<Eta>(ihcal),
                                          ch.cluster_table.get<Phi>(ihcal))
              : -1.0);
  return (dist < 0.2 ? dist : -1.0);
}

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class ECALAndHCALCaloJetLinker : public BlockElementLinkerBase {
public:
  ECALAndHCALCaloJetLinker(const edm::ParameterSet& conf)
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

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, ECALAndHCALCaloJetLinker, "ECALAndHCALCaloJetLinker");

double ECALAndHCALCaloJetLinker::testLink(size_t ielem1,
                                          size_t ielem2,
                                          reco::PFBlockElement::Type type1,
                                          reco::PFBlockElement::Type type2,
                                          const ElementListConst& elements,
                                          const PFTables& tables,
                                          const reco::PFMultiLinksIndex& multilinks) const {
  size_t ihcal_elem;
  size_t iecal_elem;

  double dist(-1.0);
  if (type1 < type2) {
    ihcal_elem = ielem1;
    iecal_elem = ielem2;
  } else {
    ihcal_elem = ielem2;
    iecal_elem = ielem1;
  }

  size_t ihcal = tables.clusters_hcal_.element_to_cluster_[ihcal_elem];
  size_t iecal = tables.clusters_ecal_.element_to_cluster_[iecal_elem];

  const auto ecal_eta = tables.clusters_ecal_.cluster_table_.get<pf::cluster::Eta>(iecal);
  const auto ecal_phi = tables.clusters_ecal_.cluster_table_.get<pf::cluster::Phi>(iecal);

  const auto hcal_eta = tables.clusters_hcal_.cluster_table_.get<pf::cluster::Eta>(ihcal);
  const auto hcal_phi = tables.clusters_hcal_.cluster_table_.get<pf::cluster::Phi>(ihcal);

  dist = LinkByRecHit::computeDist(ecal_eta, ecal_phi, hcal_eta, hcal_phi);

  return (dist < 0.2 ? dist : -1.0);
}

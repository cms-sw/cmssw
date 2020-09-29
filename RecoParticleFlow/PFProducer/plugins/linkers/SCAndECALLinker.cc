#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

#include <iostream>

using namespace edm::soa::col;

class SCAndECALLinker : public BlockElementLinkerBase {
public:
  SCAndECALLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)),
        superClusterMatchByRef_(conf.getParameter<bool>("SuperClusterMatchByRef")) {}

  double testLink(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type type1,
                  reco::PFBlockElement::Type type2,
                  const ElementListConst& elements,
                  const PFTables& tables,
                  const reco::PFMultiLinksIndex& multilinks) const override;

private:
  bool useKDTree_, debug_, superClusterMatchByRef_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, SCAndECALLinker, "SCAndECALLinker");

double SCAndECALLinker::testLink(size_t ielem1,
                                 size_t ielem2,
                                 reco::PFBlockElement::Type type1,
                                 reco::PFBlockElement::Type type2,
                                 const ElementListConst& elements,
                                 const PFTables& tables,
                                 const reco::PFMultiLinksIndex& multilinks) const {
  using Eta = pf::cluster::Eta;
  using Phi = pf::cluster::Phi;
  double dist = -1.0;

  size_t iecal_elem = 0;
  size_t isc_elem = 0;

  if (type1 < type2) {
    iecal_elem = ielem1;
    isc_elem = ielem2;
  } else {
    iecal_elem = ielem2;
    isc_elem = ielem1;
  }

  const size_t iecal = tables.clusters_ecal.element_to_cluster[iecal_elem];
  const size_t isc = tables.clusters_sc.element_to_cluster[isc_elem];

  if (superClusterMatchByRef_) {
    if (tables.clusters_sc.cluster_table.get<pf::cluster::SCRefKey>(isc) ==
        tables.clusters_ecal.cluster_table.get<pf::cluster::SCRefKey>(iecal))
      dist = 0.001;
  } else {
    //this is probably not needed any more and should be removed
    const auto& rechits_ecal = tables.clusters_ecal.cluster_to_rechit.at(iecal);
    const auto& rechits_sc = tables.clusters_sc.cluster_to_rechit.at(isc);

    if (ClusterClusterMapping::overlap(
            rechits_ecal, rechits_sc, tables.clusters_ecal.rechit_table, tables.clusters_sc.rechit_table)) {
      const auto& ct_sc = tables.clusters_sc.cluster_table;
      const auto& ct_ecal = tables.clusters_ecal.cluster_table;
      dist = LinkByRecHit::computeDist(
          ct_sc.get<Eta>(isc), ct_sc.get<Phi>(isc), ct_ecal.get<Eta>(iecal), ct_ecal.get<Phi>(iecal));
    }
  }
  return dist;
}

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm::soa::col;

class PreshowerAndECALLinker : public BlockElementLinkerBase {
public:
  PreshowerAndECALLinker(const edm::ParameterSet& conf)
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

  double testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, PreshowerAndECALLinker, "PreshowerAndECALLinker");

bool PreshowerAndECALLinker::linkPrefilter(size_t ielem1,
                                           size_t ielem2,
                                           reco::PFBlockElement::Type type1,
                                           reco::PFBlockElement::Type type2,
                                           const PFTables& tables,
                                           const reco::PFMultiLinksIndex& multilinks) const {
  bool result = false;
  // PS-ECAL KDTree multilinks are stored to PS's elem
  switch (type1) {
    case reco::PFBlockElement::PS1:
    case reco::PFBlockElement::PS2:
      result = multilinks.isLinked(ielem1, ielem2, type1, type2);
      break;
    case reco::PFBlockElement::ECAL:
      result = multilinks.isLinked(ielem2, ielem1, type2, type1);
      break;
    default:
      break;
  }
  return (useKDTree_ ? result : true);
}

double PreshowerAndECALLinker::testLink(size_t ielem1,
                                        size_t ielem2,
                                        reco::PFBlockElement::Type type1,
                                        reco::PFBlockElement::Type type2,
                                        const ElementListConst& elements,
                                        const PFTables& tables,
                                        const reco::PFMultiLinksIndex& multilinks) const {
  using Posx = pf::cluster::Posx;
  using Posy = pf::cluster::Posy;

  double dist(-1.0);

  size_t ips_elem = 0;
  size_t iecal_elem = 0;

  reco::PFBlockElement::Type type_ps;
  constexpr auto type_ecal = reco::PFBlockElement::ECAL;
  if (type1 < type2) {
    ips_elem = ielem1;
    iecal_elem = ielem2;
    type_ps = type1;
  } else {
    ips_elem = ielem2;
    iecal_elem = ielem1;
    type_ps = type2;
  }

  const auto& clusterPS = tables.getClusterTable(type_ps);
  const auto& clusterECAL = tables.getClusterTable(type_ecal);
  const size_t ips = clusterPS.element_to_cluster[ips_elem];
  const size_t iecal = clusterECAL.element_to_cluster[iecal_elem];

  // Check if the linking has been done using the KDTree algo
  // Glowinski & Gouzevitch
  if (useKDTree_) {  // KDTree algo
    const bool linked = multilinks.isLinked(ips_elem, iecal_elem, type_ps, type_ecal);

    // If the link exist, we fill dist and linktest.
    if (linked) {
      dist = LinkByRecHit::computeDist(clusterECAL.cluster_table.get<Posx>(iecal) / 1000.,
                                       clusterECAL.cluster_table.get<Posy>(iecal) / 1000.,
                                       clusterPS.cluster_table.get<Posx>(ips) / 1000.,
                                       clusterPS.cluster_table.get<Posy>(ips) / 1000.,
                                       false);
    }
  } else {  //Old algorithm
    const auto* elem1 = elements[ielem1];
    const auto* elem2 = elements[ielem2];
    dist = testLink(elem1, elem2);
  }
  return dist;
}

double PreshowerAndECALLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  const reco::PFBlockElementCluster *pselem(nullptr), *ecalelem(nullptr);
  if (elem1->type() < elem2->type()) {
    pselem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    pselem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& psref = pselem->clusterRef();
  const reco::PFClusterRef& ecalref = ecalelem->clusterRef();
  if (psref.isNull() || ecalref.isNull()) {
    throw cms::Exception("BadClusterRefs") << "PFBlockElementCluster's refs are null!";
  }

  return LinkByRecHit::testECALAndPSByRecHit(*ecalref, *psref, debug_);
}

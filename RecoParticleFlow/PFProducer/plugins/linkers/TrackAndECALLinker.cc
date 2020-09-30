#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm::soa::col;

class TrackAndECALLinker : public BlockElementLinkerBase {
public:
  TrackAndECALLinker(const edm::ParameterSet& conf)
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
  const bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndECALLinker, "TrackAndECALLinker");

bool TrackAndECALLinker::linkPrefilter(size_t ielem1,
                                       size_t ielem2,
                                       reco::PFBlockElement::Type type1,
                                       reco::PFBlockElement::Type type2,
                                       const PFTables& tables,
                                       const reco::PFMultiLinksIndex& multilinks) const {
  bool result = false;
  switch (type1) {
    case reco::PFBlockElement::TRACK:
      result = multilinks.isLinked(ielem2, ielem1, type2, type1);
      break;
    case reco::PFBlockElement::ECAL:
      result = multilinks.isValid(ielem1, ielem2, type1, type2);
    default:
      break;
  }
  return (useKDTree_ ? result : true);
}

double TrackAndECALLinker::testLink(size_t ielem1,
                                    size_t ielem2,
                                    reco::PFBlockElement::Type type1,
                                    reco::PFBlockElement::Type type2,
                                    const ElementListConst& elements,
                                    const PFTables& tables,
                                    const reco::PFMultiLinksIndex& multilinks) const {
  LogDebug("TrackAndECALLinker") << "testLink " << ielem1 << " " << ielem2;

  const auto* elem1 = elements[ielem1];
  const auto* elem2 = elements[ielem2];

  size_t itrack_elem = 0;
  size_t iecal_elem = 0;

  double dist(-1.0);
  if (type1 < type2) {
    itrack_elem = ielem1;
    iecal_elem = ielem2;
  } else {
    itrack_elem = ielem2;
    iecal_elem = ielem1;
  }
  const auto& ct_ecal = tables.clusters_ecal;
  const size_t iecal = ct_ecal.element_to_cluster[iecal_elem];
  const size_t itrack = tables.element_to_track[itrack_elem];

  // Check if the linking has been done using the KDTree algo
  // Glowinski & Gouzevitch
  if (useKDTree_) {  //KDTree Algo
    const double ecalphi = ct_ecal.cluster_table.get<pf::cluster::Phi>(iecal);
    const double ecaleta = ct_ecal.cluster_table.get<pf::cluster::Eta>(iecal);

    const bool linked =
        multilinks.isLinked(iecal_elem, itrack_elem, elements[iecal_elem]->type(), elements[itrack_elem]->type());
    // If the link exist, we fill dist and linktest.
    if (linked) {
      dist = LinkByRecHit::computeDist(ecaleta,
                                       ecalphi,
                                       tables.track_table_ecalshowermax.get<pf::track::Eta>(itrack),
                                       tables.track_table_ecalshowermax.get<pf::track::Phi>(itrack));
    }

  } else {  // Old algorithm
    dist = testLink(elem1, elem2);
  }

  LogDebug("TrackAndECALLinker") << "Linked Track-ECAL: itrack_elem=" << itrack_elem << " iecal_elem=" << iecal_elem
                                 << " dist=" << dist;

  return dist;
}

double TrackAndECALLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  double dist(-1.0);
  constexpr reco::PFTrajectoryPoint::LayerType ECALShowerMax = reco::PFTrajectoryPoint::ECALShowerMax;

  const reco::PFBlockElementCluster* ecalelem(nullptr);
  const reco::PFBlockElementTrack* tkelem(nullptr);
  if (elem1->type() < elem2->type()) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }

  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = ecalelem->clusterRef();
  const reco::PFTrajectoryPoint& tkAtECAL = trackref->extrapolatedPoint(ECALShowerMax);

  if (tkAtECAL.isValid())
    dist = LinkByRecHit::testTrackAndClusterByRecHit(*trackref, *clusterref, false, debug_);

  return dist;
}

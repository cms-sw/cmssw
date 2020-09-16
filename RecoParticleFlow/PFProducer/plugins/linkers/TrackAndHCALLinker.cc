#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace edm::soa::col;

class TrackAndHCALLinker : public BlockElementLinkerBase {
public:
  TrackAndHCALLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        trajectoryLayerEntranceString_(conf.getParameter<std::string>("trajectoryLayerEntrance")),
        trajectoryLayerExitString_(conf.getParameter<std::string>("trajectoryLayerExit")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {
    // convert TrajectoryLayers info from string to enum
    trajectoryLayerEntrance_ = reco::PFTrajectoryPoint::layerTypeByName(trajectoryLayerEntranceString_);
    trajectoryLayerExit_ = reco::PFTrajectoryPoint::layerTypeByName(trajectoryLayerExitString_);
    // make sure the requested setting is supported
    assert((trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::HCALEntrance &&
            trajectoryLayerExit_ == reco::PFTrajectoryPoint::HCALExit) ||
           (trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::HCALEntrance &&
            trajectoryLayerExit_ == reco::PFTrajectoryPoint::Unknown) ||
           (trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::VFcalEntrance &&
            trajectoryLayerExit_ == reco::PFTrajectoryPoint::Unknown));
    // flag if exit layer should be checked or not
    checkExit_ = trajectoryLayerExit_ != reco::PFTrajectoryPoint::Unknown;
  }

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

  double testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const;

private:
  bool useKDTree_;
  std::string trajectoryLayerEntranceString_;
  std::string trajectoryLayerExitString_;
  reco::PFTrajectoryPoint::LayerType trajectoryLayerEntrance_;
  reco::PFTrajectoryPoint::LayerType trajectoryLayerExit_;
  bool debug_;
  bool checkExit_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndHCALLinker, "TrackAndHCALLinker");

bool TrackAndHCALLinker::linkPrefilter(size_t ielem1,
                                       size_t ielem2,
                                       reco::PFBlockElement::Type type1,
                                       reco::PFBlockElement::Type type2,
                                       const reco::PFMultiLinksIndex& multilinks,
                                       const reco::PFBlockElement* elem1,
                                       const reco::PFBlockElement* elem2) const {
  bool result = false;
  switch (type1) {
    case reco::PFBlockElement::TRACK:
      result = multilinks.isValid(ielem1, type1, type2);
      break;
    case reco::PFBlockElement::HCAL:
      result = multilinks.isValid(ielem2, type2, type1);
    default:
      break;
  }
  return (useKDTree_ ? result : true);
}

double TrackAndHCALLinker::testLink(size_t ielem1,
                                    size_t ielem2,
                                    reco::PFBlockElement::Type type1,
                                    reco::PFBlockElement::Type type2,
                                    const ElementListConst& elements,
                                    const PFTables& tables,
                                    const reco::PFMultiLinksIndex& multilinks) const {
  const auto* elem1 = elements[ielem1];
  const auto* elem2 = elements[ielem2];
  size_t itrack_elem = 0;
  size_t ihcal_elem = 0;

  double dist(-1.0);
  if (type1 < type2) {
    itrack_elem = ielem1;
    ihcal_elem = ielem2;
  } else {
    itrack_elem = ielem2;
    ihcal_elem = ielem1;
  }
  const auto& clusterTable = tables.getClusterTable(elements[ihcal_elem]->type());
  const auto& trackTableEntrance = tables.getTrackTable(trajectoryLayerEntrance_);
  const auto& trackTableExit = checkExit_ ? tables.getTrackTable(trajectoryLayerEntrance_) : trackTableEntrance;

  size_t ihcal = clusterTable.element_to_cluster_[ihcal_elem];
  size_t itrack = tables.element_to_track_[itrack_elem];

  if (useKDTree_) {  //KDTree Algo
    const bool linked =
        multilinks.isLinked(itrack_elem, ihcal_elem, elements[itrack_elem]->type(), elements[ihcal_elem]->type());
    // If the link exist, we fill dist and linktest.
    if (linked) {
      dist = LinkByRecHit::computeTrackHCALDist(
          checkExit_, itrack, ihcal, clusterTable.cluster_table_, trackTableEntrance, trackTableExit);
    }  // multilinks verification

  } else {  // Old algorithm
    dist = testLink(elem1, elem2);
  }

  LogDebug("TrackAndHCALLinker") << "Linked Track-HCAL: itrack_elem=" << itrack_elem << " ihcal_elem=" << ihcal_elem
                                 << " dist=" << dist;
  return dist;
}

double TrackAndHCALLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  double dist(-1.0);

  const reco::PFBlockElementCluster* hcalelem(nullptr);
  const reco::PFBlockElementTrack* tkelem(nullptr);
  if (elem1->type() < elem2->type()) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }

  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = hcalelem->clusterRef();
  const reco::PFTrajectoryPoint& tkAtHCALEnt = trackref->extrapolatedPoint(trajectoryLayerEntrance_);

  if (tkAtHCALEnt.isValid())
    dist = LinkByRecHit::testTrackAndClusterByRecHit(*trackref, *clusterref, false, debug_);

  return dist;
}

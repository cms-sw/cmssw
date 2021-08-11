#include "TkPhase2OTMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

TkPhase2OTMeasurementDet::TkPhase2OTMeasurementDet(const GeomDet* gdet, Phase2OTMeasurementConditionSet& conditions)
    : MeasurementDet(gdet), theDetConditions(&conditions) {
  if (dynamic_cast<const PixelGeomDetUnit*>(gdet) == nullptr) {
    throw MeasurementDetException(
        "TkPhase2OTMeasurementDet constructed with a GeomDet which is not a PixelGeomDetUnit");
  }
}

bool TkPhase2OTMeasurementDet::measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                                            const MeasurementEstimator& est,
                                            const MeasurementTrackerEvent& data,
                                            TempMeasurements& result) const {
  if (!isActive(data)) {
    result.add(theInactiveHit, 0.F);
    return true;
  }

  if (recHits(stateOnThisDet, est, data, result.hits, result.distances))
    return true;

  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
  bool inac = hasBadComponents(stateOnThisDet, data);
  result.add(inac ? theInactiveHit : theMissingHit, 0.F);
  return inac;
}

bool TkPhase2OTMeasurementDet::recHits(const TrajectoryStateOnSurface& stateOnThisDet,
                                       const MeasurementEstimator& est,
                                       const MeasurementTrackerEvent& data,
                                       RecHitContainer& result,
                                       std::vector<float>& diffs) const {
  if UNLIKELY ((!isActive(data)) || isEmpty(data.phase2OTData()))
    return false;

  auto oldSize = result.size();

  const detset& detSet = data.phase2OTData().detSet(index());
  auto begin = &(data.phase2OTData().handle()->data().front());
  auto reject = [&](auto ci) -> bool {
    return (!data.phase2OTClustersToSkip().empty()) && data.phase2OTClustersToSkip()[ci - begin];
  };

  /// use the usual 5 sigma cut from the Traj to identify the column....
  auto firstCluster = detSet.begin();
  auto lastCluster = detSet.end();

  // do not use this as it does not account for APE...
  // auto xyLimits = est.maximalLocalDisplacement(stateOnThisDet,fastGeomDet().specificSurface());
  auto le = stateOnThisDet.localError().positionError();
  LocalError lape = static_cast<TrackerGeomDet const&>(fastGeomDet()).localAlignmentError();
  auto ye = le.yy();
  if (lape.valid()) {
    ye += lape.yy();
  }
  // 5 sigma to be on the safe side
  ye = 5.f * std::sqrt(ye);
  LocalVector maxD(0, ye, 0);
  // pixel topology is rectangular: x and y are independent
  auto ymin = specificGeomDet().specificTopology().measurementPosition(stateOnThisDet.localPosition() - maxD);
  auto ymax = specificGeomDet().specificTopology().measurementPosition(stateOnThisDet.localPosition() + maxD);
  int utraj = ymin.x();
  // do not apply for iteration not cutting on propagation
  if (est.maxSagitta() >= 0) {
    int colMin = ymin.y();
    int colMax = ymax.y();
    firstCluster = std::find_if(firstCluster, detSet.end(), [colMin](const Phase2TrackerCluster1D& hit) {
      return int(hit.column()) >= colMin;
    });
    lastCluster = std::find_if(
        firstCluster, detSet.end(), [colMax](const Phase2TrackerCluster1D& hit) { return int(hit.column()) > colMax; });
  }

  while (firstCluster != lastCluster) {  // loop on each column
    auto const col = firstCluster->column();
    auto endCluster = std::find_if(
        firstCluster, detSet.end(), [col](const Phase2TrackerCluster1D& hit) { return hit.column() != col; });
    // find trajectory position in this column
    auto rightCluster = std::find_if(
        firstCluster, endCluster, [utraj](const Phase2TrackerCluster1D& hit) { return int(hit.firstStrip()) > utraj; });
    // search for compatible clusters...
    if (rightCluster != firstCluster) {
      // there are hits on the left of the utraj
      auto leftCluster = rightCluster;
      while (--leftCluster >= firstCluster) {
        if (reject(leftCluster))
          continue;
        Phase2TrackerCluster1DRef cluster = detSet.makeRefTo(data.phase2OTData().handle(), leftCluster);
        auto hit = buildRecHit(cluster, stateOnThisDet.localParameters());
        auto diffEst = est.estimate(stateOnThisDet, *hit);
        if (!diffEst.first)
          break;  // exit loop on first incompatible hit
        result.push_back(hit);
        diffs.push_back(diffEst.second);
      }
    }
    for (; rightCluster != endCluster; rightCluster++) {
      if (reject(rightCluster))
        continue;
      Phase2TrackerCluster1DRef cluster = detSet.makeRefTo(data.phase2OTData().handle(), rightCluster);
      auto hit = buildRecHit(cluster, stateOnThisDet.localParameters());
      auto diffEst = est.estimate(stateOnThisDet, *hit);
      if (!diffEst.first)
        break;  // exit loop on first incompatible hit
      result.push_back(hit);
      diffs.push_back(diffEst.second);
    }
    firstCluster = endCluster;
  }  // loop over columns
  return result.size() > oldSize;
}

TrackingRecHit::RecHitPointer TkPhase2OTMeasurementDet::buildRecHit(const Phase2TrackerCluster1DRef& cluster,
                                                                    const LocalTrajectoryParameters& ltp) const {
  const PixelGeomDetUnit& gdu(specificGeomDet());
  auto&& params = cpe()->localParameters(*cluster, gdu);

  return std::make_shared<Phase2TrackerRecHit1D>(params.first, params.second, fastGeomDet(), cluster);
}

TkPhase2OTMeasurementDet::RecHitContainer TkPhase2OTMeasurementDet::recHits(const TrajectoryStateOnSurface& ts,
                                                                            const MeasurementTrackerEvent& data) const {
  RecHitContainer result;
  if (isEmpty(data.phase2OTData()))
    return result;
  if (!isActive(data))
    return result;
  const Phase2TrackerCluster1D* begin = nullptr;
  if (!data.phase2OTData().handle()->data().empty()) {
    begin = &(data.phase2OTData().handle()->data().front());
  }
  const detset& detSet = data.phase2OTData().detSet(index());
  result.reserve(detSet.size());
  for (const_iterator ci = detSet.begin(); ci != detSet.end(); ++ci) {
    if (ci < begin) {
      edm::LogError("IndexMisMatch") << "TkPhase2OTMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
    unsigned int index = ci - begin;
    if (!data.phase2OTClustersToSkip().empty() && index >= data.phase2OTClustersToSkip().size()) {
      edm::LogError("IndexMisMatch") << "TkPhase2OTMeasurementDet cannot create hit because of index mismatch. i.e "
                                     << index << " >= " << data.phase2OTClustersToSkip().size();
      return result;
    }
    if (data.phase2OTClustersToSkip().empty() or (not data.phase2OTClustersToSkip()[index])) {
      Phase2TrackerCluster1DRef cluster = detSet.makeRefTo(data.phase2OTData().handle(), ci);
      result.push_back(buildRecHit(cluster, ts.localParameters()));
    } else {
      LogDebug("TkPhase2OTMeasurementDet") << "skipping this cluster from last iteration on "
                                           << fastGeomDet().geographicalId().rawId() << " key: " << index;
    }
  }
  return result;
}

//FIXME:just temporary solution for phase2!
bool TkPhase2OTMeasurementDet::hasBadComponents(const TrajectoryStateOnSurface& tsos,
                                                const MeasurementTrackerEvent& data) const {
  /*
    if (badRocPositions_.empty()) return false;
    LocalPoint lp = tsos.localPosition();
    LocalError le = tsos.localError().positionError();
    double dx = 3*std::sqrt(le.xx()) + theRocWidth, dy = 3*std::sqrt(le.yy()) + theRocHeight;
    for (std::vector<LocalPoint>::const_iterator it = badRocPositions_.begin(), ed = badRocPositions_.end(); it != ed; ++it) {
        if ( (std::abs(it->x() - lp.x()) < dx) &&
             (std::abs(it->y() - lp.y()) < dy) ) return true;
    } 
*/
  return false;
}

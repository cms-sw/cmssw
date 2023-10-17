#include "TkPixelMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

namespace {
  // in cms units are in cm
  constexpr float theRocWidth = 0.81 / 2;
  constexpr float theRocHeight = 0.81 / 2;
}  // namespace

TkPixelMeasurementDet::TkPixelMeasurementDet(const GeomDet* gdet, PxMeasurementConditionSet& conditions)
    : MeasurementDet(gdet), index_(0), theDetConditions(&conditions) {
  if (dynamic_cast<const PixelGeomDetUnit*>(gdet) == nullptr) {
    throw MeasurementDetException("TkPixelMeasurementDet constructed with a GeomDet which is not a PixelGeomDetUnit");
  }
}

bool TkPixelMeasurementDet::measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                                         const MeasurementEstimator& est,
                                         const MeasurementTrackerEvent& data,
                                         TempMeasurements& result) const {
  if (!isActive(data)) {
    result.add(theInactiveHit, 0.F);
    return true;
  }

  auto xl = 100.f;
  auto yl = 100.f;
  // do not apply for iteration not cutting on propagation
  if (est.maxSagitta() >= 0) {
    // do not use this as it does not account for APE...
    // auto xyLimits = est.maximalLocalDisplacement(stateOnThisDet,fastGeomDet().specificSurface());
    auto le = stateOnThisDet.localError().positionError();
    LocalError lape = static_cast<TrackerGeomDet const&>(fastGeomDet()).localAlignmentError();
    xl = le.xx();
    yl = le.yy();
    if (lape.valid()) {
      xl += lape.xx();
      yl += lape.yy();
    }
    // 5 sigma to be on the safe side
    xl = 5.f * std::sqrt(xl);
    yl = 5.f * std::sqrt(yl);
  }

  auto oldSize = result.size();
  MeasurementDet::RecHitContainer&& allHits = compHits(stateOnThisDet, data, xl, yl);
  for (auto&& hit : allHits) {
    std::pair<bool, double> diffEst = est.estimate(stateOnThisDet, *hit);
    if (diffEst.first)
      result.add(std::move(hit), diffEst.second);
  }

  if (result.size() > oldSize)
    return true;

  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
  bool inac = hasBadComponents(stateOnThisDet, data);
  result.add(inac ? theInactiveHit : theMissingHit, 0.F);
  return inac;
}

TrackingRecHit::RecHitPointer TkPixelMeasurementDet::buildRecHit(const SiPixelClusterRef& cluster,
                                                                 const LocalTrajectoryParameters& ltp) const {
  const GeomDetUnit& gdu(specificGeomDet());

  auto&& params = cpe()->getParameters(*cluster, gdu, ltp);
  return std::make_shared<SiPixelRecHit>(
      std::get<0>(params), std::get<1>(params), std::get<2>(params), fastGeomDet(), cluster);
}

TkPixelMeasurementDet::RecHitContainer TkPixelMeasurementDet::recHits(const TrajectoryStateOnSurface& ts,
                                                                      const MeasurementTrackerEvent& data) const {
  float xl = 100.f;  // larger than any detector
  float yl = 100.f;
  return compHits(ts, data, xl, yl);
}

TkPixelMeasurementDet::RecHitContainer TkPixelMeasurementDet::compHits(const TrajectoryStateOnSurface& ts,
                                                                       const MeasurementTrackerEvent& data,
                                                                       float xl,
                                                                       float yl) const {
  RecHitContainer result;
  if (isEmpty(data.pixelData()) == true)
    return result;
  if (isActive(data) == false)
    return result;
  const SiPixelCluster* begin = nullptr;
  if (!data.pixelData().handle()->data().empty()) {
    begin = &(data.pixelData().handle()->data().front());
  }
  const detset& detSet = data.pixelData().detSet(index());
  result.reserve(detSet.size());

  // pixel topology is rectangular, all positions are independent
  LocalVector maxD(xl, yl, 0);
  auto PMinus = specificGeomDet().specificTopology().measurementPosition(ts.localPosition() - maxD);
  auto PPlus = specificGeomDet().specificTopology().measurementPosition(ts.localPosition() + maxD);

  int xminus = PMinus.x();
  int yminus = PMinus.y();
  int xplus = PPlus.x() + 0.5f;
  int yplus = PPlus.y() + 0.5f;

  // rechits are sorted in x...
  auto rightCluster = std::find_if(
      detSet.begin(), detSet.end(), [xplus](const SiPixelCluster& cl) { return cl.minPixelRow() > xplus; });

  // std::cout << "px xlim " << xl << ' ' << xminus << '/' << xplus << ' ' << rightCluster-detSet.begin() << ',' << detSet.end()-rightCluster << std::endl;

  // consider only compatible clusters
  for (auto ci = detSet.begin(); ci != rightCluster; ++ci) {
    if (ci < begin) {
      edm::LogError("IndexMisMatch") << "TkPixelMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
    unsigned int index = ci - begin;
    if (!data.pixelClustersToSkip().empty() && index >= data.pixelClustersToSkip().size()) {
      edm::LogError("IndexMisMatch") << "TkPixelMeasurementDet cannot create hit because of index mismatch. i.e "
                                     << index << " >= " << data.pixelClustersToSkip().size();
      return result;
    }

    if (ci->maxPixelRow() < xminus)
      continue;
    // also check compatibility in y... (does not add much)
    if (ci->minPixelCol() > yplus)
      continue;
    if (ci->maxPixelCol() < yminus)
      continue;

    if (data.pixelClustersToSkip().empty() or (not data.pixelClustersToSkip()[index])) {
      SiPixelClusterRef cluster = detSet.makeRefTo(data.pixelData().handle(), ci);
      result.push_back(buildRecHit(cluster, ts.localParameters()));
    } else {
      LogDebug("TkPixelMeasurementDet") << "skipping this cluster from last iteration on "
                                        << fastGeomDet().geographicalId().rawId() << " key: " << index;
    }
  }
  return result;
}

bool TkPixelMeasurementDet::hasBadComponents(const TrajectoryStateOnSurface& tsos,
                                             const MeasurementTrackerEvent& data) const {
  auto badFEDChannelPositions = getBadFEDChannelPositions(data);
  if (badRocPositions_.empty() && badFEDChannelPositions == nullptr)
    return false;

  auto lp = tsos.localPosition();
  auto le = tsos.localError().positionError();
  for (auto const& broc : badRocPositions_) {
    auto dx = std::abs(broc.x() - lp.x()) - theRocWidth;
    auto dy = std::abs(broc.y() - lp.y()) - theRocHeight;
    if ((dx <= 0.f) & (dy <= 0.f))
      return true;
    if ((dx * dx < 9.f * le.xx()) && (dy * dy < 9.f * le.yy()))
      return true;
  }

  if (badFEDChannelPositions == nullptr)
    return false;
  float dx = 3.f * std::sqrt(le.xx()) + theRocWidth, dy = 3.f * std::sqrt(le.yy()) + theRocHeight;
  for (auto const& p : *badFEDChannelPositions) {
    if (lp.x() > (p.first.x() - dx) && lp.x() < (p.second.x() + dx) && lp.y() > (p.first.y() - dy) &&
        lp.y() < (p.second.y() + dy)) {
      return true;
    }
  }

  return false;
}

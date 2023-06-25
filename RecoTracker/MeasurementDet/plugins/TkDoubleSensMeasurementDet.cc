#include "TkDoubleSensMeasurementDet.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

using namespace std;

TkDoubleSensMeasurementDet::TkDoubleSensMeasurementDet(const DoubleSensGeomDet* gdet,
                                                       const PixelClusterParameterEstimator* cpe)
    : MeasurementDet(gdet), thePixelCPE(cpe), theFirstDet(nullptr), theSecondDet(nullptr) {}

void TkDoubleSensMeasurementDet::init(const MeasurementDet* firstDet, const MeasurementDet* secondDet) {
  theFirstDet = dynamic_cast<const TkPixelMeasurementDet*>(firstDet);
  theSecondDet = dynamic_cast<const TkPixelMeasurementDet*>(secondDet);

  if ((theFirstDet == nullptr) || (theSecondDet == nullptr)) {
    throw MeasurementDetException(
        "TkDoubleSensMeasurementDet ERROR: Trying to glue a det which is not a TkPixelMeasurementDet");
  }
}

TkDoubleSensMeasurementDet::RecHitContainer TkDoubleSensMeasurementDet::recHits(
    const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent& data) const {
  RecHitContainer result;

  if (data.pixelData().handle()->data().empty())
    return result;
  LogTrace("MeasurementTracker") << " is not empty";
  if (!isActive(data))
    return result;
  LogTrace("MeasurementTracker") << " and is active";

  //find clusters to skip
  const detset& firstDetSet = data.pixelData().detSet(firstDet()->index());
  const detset& secondDetSet = data.pixelData().detSet(secondDet()->index());
  std::vector<bool> skipClustersUpper(data.pixelClustersToSkip().empty() ? 0 : secondDetSet.size(), false);
  std::vector<bool> skipClustersLower(data.pixelClustersToSkip().empty() ? 0 : firstDetSet.size(), false);

  const SiPixelCluster* begin = nullptr;
  if (!data.pixelData().handle()->data().empty()) {
    begin = &(data.pixelData().handle()->data().front());
  }
  if (!data.pixelClustersToSkip().empty()) {
    if (!firstDetSet.empty()) {
      for (const_iterator cil = firstDetSet.begin(); cil != firstDetSet.end(); ++cil) {
        if (cil < begin) {
          edm::LogError("IndexMisMatch") << "TkDoubleSensMeasurementDet cannot create hit because of index mismatch.";
          return result;
        }
        unsigned int indexl = cil - begin;
        if (data.pixelClustersToSkip()[indexl]) {
          int iLocalL = std::distance(firstDetSet.begin(), cil);
          skipClustersLower[iLocalL] = true;
        }
      }
    }
    if (!secondDetSet.empty()) {
      for (const_iterator ciu = secondDetSet.begin(); ciu != secondDetSet.end(); ++ciu) {
        if (ciu < begin) {
          edm::LogError("IndexMisMatch") << "TkDoubleSensMeasurementDet cannot create hit because of index mismatch.";
          return result;
        }
        unsigned int indexu = ciu - begin;
        if (data.pixelClustersToSkip()[indexu]) {
          int iLocalU = std::distance(secondDetSet.begin(), ciu);
          skipClustersUpper[iLocalU] = true;
        }
      }
    }
  }

  return result;
}

bool TkDoubleSensMeasurementDet::measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                                              const MeasurementEstimator& est,
                                              const MeasurementTrackerEvent& data,
                                              TempMeasurements& result) const {
  LogDebug("MeasurementTracker") << "TkDoubleSensMeasurementDet::measurements";

  if (!isActive(data)) {
    result.add(theInactiveHit, 0.F);
    return true;
  }

  LogTrace("MeasurementTracker") << " is active";

  auto oldSize = result.size();
  MeasurementDet::RecHitContainer&& allHits = recHits(stateOnThisDet, data);

  for (auto&& hit : allHits) {
    std::pair<bool, double> diffEst = est.estimate(stateOnThisDet, *hit);
    if (diffEst.first) {
      LogDebug("MeasurementTracker") << "New vh added with chi2: " << diffEst.second;
      result.add(std::move(hit), diffEst.second);
    }
  }

  if (result.size() > oldSize)
    return true;

  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
  result.add(theMissingHit, 0.F);
  LogDebug("MeasurementTracker") << "adding missing hit";
  return false;
}

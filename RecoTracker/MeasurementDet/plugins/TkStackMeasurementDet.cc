#include "TkStackMeasurementDet.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

using namespace std;

TkStackMeasurementDet::TkStackMeasurementDet(const StackGeomDet* gdet, const PixelClusterParameterEstimator* cpe)
    : MeasurementDet(gdet), thePixelCPE(cpe), theLowerDet(nullptr), theUpperDet(nullptr) {}

void TkStackMeasurementDet::init(const MeasurementDet* lowerDet, const MeasurementDet* upperDet) {
  theLowerDet = dynamic_cast<const TkPhase2OTMeasurementDet*>(lowerDet);
  theUpperDet = dynamic_cast<const TkPhase2OTMeasurementDet*>(upperDet);

  if ((theLowerDet == nullptr) || (theUpperDet == nullptr)) {
    throw MeasurementDetException(
        "TkStackMeasurementDet ERROR: Trying to glue a det which is not a TkPhase2OTMeasurementDet");
  }
}

TkStackMeasurementDet::RecHitContainer TkStackMeasurementDet::recHits(const TrajectoryStateOnSurface& ts,
                                                                      const MeasurementTrackerEvent& data) const {
  RecHitContainer result;

  if (data.phase2OTVectorHits().empty())
    return result;
  LogTrace("MeasurementTracker") << " is not empty";
  if (!isActive(data))
    return result;
  LogTrace("MeasurementTracker") << " and is active";

  //find clusters to skip
  const detset& lowerDetSet = data.phase2OTData().detSet(lowerDet()->index());
  const detset& upperDetSet = data.phase2OTData().detSet(upperDet()->index());
  std::vector<bool> skipClustersUpper(data.phase2OTClustersToSkip().empty() ? 0 : upperDetSet.size(), false);
  std::vector<bool> skipClustersLower(data.phase2OTClustersToSkip().empty() ? 0 : lowerDetSet.size(), false);

  const Phase2TrackerCluster1D* begin = nullptr;
  if (!data.phase2OTData().handle()->data().empty()) {
    begin = &(data.phase2OTData().handle()->data().front());
  }
  if (!data.phase2OTClustersToSkip().empty()) {
    if (!lowerDetSet.empty()) {
      for (const_iterator cil = lowerDetSet.begin(); cil != lowerDetSet.end(); ++cil) {
        if (cil < begin) {
          edm::LogError("IndexMisMatch") << "TkStackMeasurementDet cannot create hit because of index mismatch.";
          return result;
        }
        unsigned int indexl = cil - begin;
        if (data.phase2OTClustersToSkip()[indexl]) {
          int iLocalL = std::distance(lowerDetSet.begin(), cil);
          skipClustersLower[iLocalL] = true;
        }
      }
    }
    if (!upperDetSet.empty()) {
      for (const_iterator ciu = upperDetSet.begin(); ciu != upperDetSet.end(); ++ciu) {
        if (ciu < begin) {
          edm::LogError("IndexMisMatch") << "TkStackMeasurementDet cannot create hit because of index mismatch.";
          return result;
        }
        unsigned int indexu = ciu - begin;
        if (data.phase2OTClustersToSkip()[indexu]) {
          int iLocalU = std::distance(upperDetSet.begin(), ciu);
          skipClustersUpper[iLocalU] = true;
        }
      }
    }
  }
  DetId detIdStack = specificGeomDet().geographicalId();

  auto iterator = data.phase2OTVectorHits().find(detIdStack);
  if (iterator == data.phase2OTVectorHits().end())
    return result;
  for (const auto& vecHit : data.phase2OTVectorHits()[detIdStack]) {
    if (!data.phase2OTClustersToSkip().empty()) {
      if (skipClustersLower[vecHit.lowerCluster().key() - lowerDetSet.offset()])
        continue;
      if (skipClustersUpper[vecHit.upperCluster().key() - upperDetSet.offset()])
        continue;
    }
    result.push_back(std::make_shared<VectorHit>(vecHit));
  }

  iterator = data.phase2OTVectorHitsRej().find(detIdStack);
  if (iterator == data.phase2OTVectorHitsRej().end())
    return result;
  for (const auto& vecHit : data.phase2OTVectorHitsRej()[detIdStack]) {
    if (!data.phase2OTClustersToSkip().empty()) {
      if (skipClustersLower[vecHit.lowerCluster().key() - lowerDetSet.offset()])
        continue;
      if (skipClustersUpper[vecHit.upperCluster().key() - upperDetSet.offset()])
        continue;
    }
    result.push_back(std::make_shared<VectorHit>(vecHit));
  }

  return result;
}

bool TkStackMeasurementDet::measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                                         const MeasurementEstimator& est,
                                         const MeasurementTrackerEvent& data,
                                         TempMeasurements& result) const {
  LogDebug("MeasurementTracker") << "TkStackMeasurementDet::measurements";

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

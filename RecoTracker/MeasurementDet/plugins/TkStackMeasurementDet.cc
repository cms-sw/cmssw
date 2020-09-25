#include "TkStackMeasurementDet.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"

using namespace std;

TkStackMeasurementDet::TkStackMeasurementDet(const StackGeomDet* gdet,
                                             const VectorHitBuilderAlgorithm* matcher,
                                             const PixelClusterParameterEstimator* cpe)
    : MeasurementDet(gdet), theMatcher(matcher), thePixelCPE(cpe), theLowerDet(nullptr), theUpperDet(nullptr) {}

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

  // Old solution creating the VHs on the fly. Keep for now
  /*
  const Phase2TrackerCluster1D* begin = nullptr;
  if (!data.phase2OTData().handle()->data().empty()) {
    begin = &(data.phase2OTData().handle()->data().front());
  }

  LogTrace("MeasurementTracker") << "TkStackMeasurementDet::recHits algo has been set" << std::endl;

  const detset& lowerDetSet = data.phase2OTData().detSet(lowerDet()->index());
  const detset& upperDetSet = data.phase2OTData().detSet(upperDet()->index());

  LogTrace("MeasurementTracker") << " DetSets set with sizes:" << lowerDetSet.size() << " and " << upperDetSet.size()
                                 << "!";
  result.reserve(lowerDetSet.size() > upperDetSet.size() ? lowerDetSet.size() : upperDetSet.size());

  for (const_iterator cil = lowerDetSet.begin(); cil != lowerDetSet.end(); ++cil) {
    if (cil < begin) {
      edm::LogError("IndexMisMatch") << "TkStackMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
    unsigned int indexl = cil - begin;
    LogTrace("MeasurementTracker") << " index cluster lower" << indexl << " on detId "
                                   << fastGeomDet().geographicalId().rawId();

    for (const_iterator ciu = upperDetSet.begin(); ciu != upperDetSet.end(); ++ciu) {
      unsigned int indexu = ciu - begin;
      if (ciu < begin) {
        edm::LogError("IndexMisMatch") << "TkStackMeasurementDet cannot create hit because of index mismatch.";
        return result;
      }
      LogTrace("VectorHitBuilderAlgorithm") << " index cluster upper " << indexu;

      if (data.phase2OTClustersToSkip().empty() or
          ((not data.phase2OTClustersToSkip()[indexl]) and (not data.phase2OTClustersToSkip()[indexu]))) {
        Phase2TrackerCluster1DRef clusterLower = edmNew::makeRefTo(data.phase2OTData().handle(), cil);
        Phase2TrackerCluster1DRef clusterUpper = edmNew::makeRefTo(data.phase2OTData().handle(), ciu);
        //ERICA:I would have prefer to keep buildVectorHits ...
        VectorHit vh = theMatcher->buildVectorHit(&specificGeomDet(), clusterLower, clusterUpper);
        LogTrace("MeasurementTracker") << "TkStackMeasurementDet::rechits adding VectorHits!" << std::endl;
        LogTrace("MeasurementTracker") << vh << std::endl;
        result.push_back(std::make_shared<VectorHit>(vh));
      }
    }
  }
  

*/

  //find clusters to skip
  std::vector<Phase2TrackerCluster1DRef> skipClustersLower;
  std::vector<Phase2TrackerCluster1DRef> skipClustersUpper;
  const Phase2TrackerCluster1D* begin = nullptr;
  if (!data.phase2OTData().handle()->data().empty()) {
    begin = &(data.phase2OTData().handle()->data().front());
  }
  if (!data.phase2OTClustersToSkip().empty()) {
    const detset& lowerDetSet = data.phase2OTData().detSet(lowerDet()->index());
    const detset& upperDetSet = data.phase2OTData().detSet(upperDet()->index());

    if (!lowerDetSet.empty()) {
      for (const_iterator cil = lowerDetSet.begin(); cil != lowerDetSet.end(); ++cil) {
        if (cil < begin) {
          edm::LogError("IndexMisMatch") << "TkStackMeasurementDet cannot create hit because of index mismatch.";
          return result;
        }
        unsigned int indexl = cil - begin;
        if (data.phase2OTClustersToSkip()[indexl]) {
          Phase2TrackerCluster1DRef clusterRef = edmNew::makeRefTo(data.phase2OTData().handle(), cil);
          skipClustersLower.push_back(clusterRef);
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
          Phase2TrackerCluster1DRef clusterRef = edmNew::makeRefTo(data.phase2OTData().handle(), ciu);
          skipClustersUpper.push_back(clusterRef);
        }
      }
    }
  }

  DetId detIdStack = specificGeomDet().geographicalId();

  auto iterator = data.phase2OTVectorHits().find(detIdStack);
  if (iterator == data.phase2OTVectorHits().end())
    return result;
  for (const auto& vecHit : data.phase2OTVectorHits()[detIdStack]) {
    if (std::find(skipClustersLower.begin(), skipClustersLower.end(), vecHit.lowerCluster()) != skipClustersLower.end())
      continue;
    if (std::find(skipClustersUpper.begin(), skipClustersUpper.end(), vecHit.upperCluster()) != skipClustersUpper.end())
      continue;
    result.push_back(std::make_shared<VectorHit>(vecHit));
  }

  iterator = data.phase2OTVectorHitsRej().find(detIdStack);
  if (iterator == data.phase2OTVectorHitsRej().end())
    return result;
  for (const auto& vecHit : data.phase2OTVectorHitsRej()[detIdStack]) {
    if (std::find(skipClustersLower.begin(), skipClustersLower.end(), vecHit.lowerCluster()) != skipClustersLower.end())
      continue;
    if (std::find(skipClustersUpper.begin(), skipClustersUpper.end(), vecHit.upperCluster()) != skipClustersUpper.end())
      continue;
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

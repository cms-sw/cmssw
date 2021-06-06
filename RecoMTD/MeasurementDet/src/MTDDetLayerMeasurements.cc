/** \class MTDDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  \author B. Tannenwald 
 *  Adapted from RecoMuon version.
 *
 */

#include "RecoMTD/MeasurementDet/interface/MTDDetLayerMeasurements.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

typedef std::shared_ptr<GenericTransientTrackingRecHit> MTDRecHitPointer;
typedef std::vector<GenericTransientTrackingRecHit::RecHitPointer> MTDRecHitContainer;
typedef MTDDetLayerMeasurements::MeasurementContainer MeasurementContainer;

MTDDetLayerMeasurements::MTDDetLayerMeasurements(const edm::InputTag& mtdlabel, edm::ConsumesCollector& iC)
    : theMTDToken(iC.consumes<MTDTrackingRecHit>(mtdlabel)),
      theMTDRecHits(),
      theMTDEventCacheID(0),
      theEvent(nullptr) {}

MTDDetLayerMeasurements::~MTDDetLayerMeasurements() {}

MTDRecHitContainer MTDDetLayerMeasurements::recHits(const GeomDet* geomDet, const edm::Event& iEvent) {
  DetId geoId = geomDet->geographicalId();
  theEvent = &iEvent;
  MTDRecHitContainer result;

  checkMTDRecHits();

  // Create the ChamberId
  DetId detId(geoId.rawId());
  LogDebug("MTDDetLayerMeasurements") << "(MTD): " << static_cast<MTDDetId>(detId) << std::endl;

  // Get the MTD-Segment which relies on this chamber
  auto detset = (*theMTDRecHits)[detId];

  for (const auto& rechit : detset)
    result.push_back(GenericTransientTrackingRecHit::build(geomDet, &rechit));

  return result;
}

void MTDDetLayerMeasurements::checkMTDRecHits() {
  LogDebug("MTDDetLayerMeasurements") << "Checking MTD RecHits";
  checkEvent();
  auto cacheID = theEvent->cacheIdentifier();
  if (cacheID == theMTDEventCacheID)
    return;

  {
    theEvent->getByToken(theMTDToken, theMTDRecHits);
    theMTDEventCacheID = cacheID;
  }
  if (!theMTDRecHits.isValid()) {
    throw cms::Exception("MTDDetLayerMeasurements") << "Cannot get MTD RecHits";
  }
}

template <class T>
T MTDDetLayerMeasurements::sortResult(T& result) {
  if (!result.empty()) {
    sort(result.begin(), result.end(), TrajMeasLessEstim());
  }

  return result;
}

///measurements method if already got the Event
MeasurementContainer MTDDetLayerMeasurements::measurements(const DetLayer* layer,
                                                           const TrajectoryStateOnSurface& startingState,
                                                           const Propagator& prop,
                                                           const MeasurementEstimator& est) {
  checkEvent();
  return measurements(layer, startingState, prop, est, *theEvent);
}

MeasurementContainer MTDDetLayerMeasurements::measurements(const DetLayer* layer,
                                                           const TrajectoryStateOnSurface& startingState,
                                                           const Propagator& prop,
                                                           const MeasurementEstimator& est,
                                                           const edm::Event& iEvent) {
  MeasurementContainer result;

  const auto& dss = layer->compatibleDets(startingState, prop, est);
  LogDebug("MTDDetLayerMeasurements") << "compatibleDets: " << dss.size() << std::endl;

  for (const auto& dws : dss) {
    MeasurementContainer detMeasurements = measurements(layer, dws.first, dws.second, est, iEvent);
    result.insert(result.end(), detMeasurements.begin(), detMeasurements.end());
  }

  return sortResult(result);
}

MeasurementContainer MTDDetLayerMeasurements::measurements(const DetLayer* layer,
                                                           const GeomDet* det,
                                                           const TrajectoryStateOnSurface& stateOnDet,
                                                           const MeasurementEstimator& est,
                                                           const edm::Event& iEvent) {
  MeasurementContainer result;

  // Get the Segments which relies on the GeomDet given by compatibleDets
  MTDRecHitContainer mtdRecHits = recHits(det, iEvent);

  // Create the Trajectory Measurement
  for (const auto& rechit : mtdRecHits) {
    MeasurementEstimator::HitReturnType estimate = est.estimate(stateOnDet, *rechit);
    LogDebug("RecoMTD") << "Dimension: " << rechit->dimension() << " Chi2: " << estimate.second << std::endl;
    if (estimate.first) {
      result.push_back(TrajectoryMeasurement(stateOnDet, rechit, estimate.second, layer));
    }
  }

  return sortResult(result);
}

MeasurementContainer MTDDetLayerMeasurements::fastMeasurements(const DetLayer* layer,
                                                               const TrajectoryStateOnSurface& theStateOnDet,
                                                               const TrajectoryStateOnSurface& startingState,
                                                               const Propagator& prop,
                                                               const MeasurementEstimator& est,
                                                               const edm::Event& iEvent) {
  MeasurementContainer result;
  MTDRecHitContainer rhs = recHits(layer, iEvent);
  for (const auto& irh : rhs) {
    MeasurementEstimator::HitReturnType estimate = est.estimate(theStateOnDet, (*irh));
    if (estimate.first) {
      result.push_back(TrajectoryMeasurement(theStateOnDet, irh, estimate.second, layer));
    }
  }

  return sortResult(result);
}

///fastMeasurements method if already got the Event
MeasurementContainer MTDDetLayerMeasurements::fastMeasurements(const DetLayer* layer,
                                                               const TrajectoryStateOnSurface& theStateOnDet,
                                                               const TrajectoryStateOnSurface& startingState,
                                                               const Propagator& prop,
                                                               const MeasurementEstimator& est) {
  checkEvent();
  return fastMeasurements(layer, theStateOnDet, startingState, prop, est, *theEvent);
}

std::vector<TrajectoryMeasurementGroup> MTDDetLayerMeasurements::groupedMeasurements(
    const DetLayer* layer,
    const TrajectoryStateOnSurface& startingState,
    const Propagator& prop,
    const MeasurementEstimator& est) {
  checkEvent();
  return groupedMeasurements(layer, startingState, prop, est, *theEvent);
}

std::vector<TrajectoryMeasurementGroup> MTDDetLayerMeasurements::groupedMeasurements(
    const DetLayer* layer,
    const TrajectoryStateOnSurface& startingState,
    const Propagator& prop,
    const MeasurementEstimator& est,
    const edm::Event& iEvent) {
  std::vector<TrajectoryMeasurementGroup> result;
  // if we want to use the concept of InvalidRecHits,
  // we can reuse LayerMeasurements from TrackingTools/MeasurementDet
  std::vector<DetGroup> groups(layer->groupedCompatibleDets(startingState, prop, est));

  for (const auto& grp : groups) {
    std::vector<TrajectoryMeasurement> groupMeasurements;
    for (const auto& detAndStateItr : grp) {
      std::vector<TrajectoryMeasurement> detMeasurements =
          measurements(layer, detAndStateItr.det(), detAndStateItr.trajectoryState(), est, iEvent);
      groupMeasurements.insert(groupMeasurements.end(), detMeasurements.begin(), detMeasurements.end());
    }

    result.push_back(TrajectoryMeasurementGroup(sortResult(groupMeasurements), grp));
  }

  return result;
}

///set event
void MTDDetLayerMeasurements::setEvent(const edm::Event& event) { theEvent = &event; }

void MTDDetLayerMeasurements::checkEvent() const {
  if (!theEvent)
    throw cms::Exception("MTDDetLayerMeasurements") << "The event has not been set";
}

MTDRecHitContainer MTDDetLayerMeasurements::recHits(const DetLayer* layer, const edm::Event& iEvent) {
  MTDRecHitContainer rhs;

  std::vector<const GeomDet*> gds = layer->basicComponents();

  for (const GeomDet* igd : gds) {
    MTDRecHitContainer detHits = recHits(igd, iEvent);
    rhs.insert(rhs.end(), detHits.begin(), detHits.end());
  }
  return rhs;
}

MTDRecHitContainer MTDDetLayerMeasurements::recHits(const DetLayer* layer) {
  checkEvent();
  return recHits(layer, *theEvent);
}

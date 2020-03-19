#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

TrajectorySeedHitCandidate::TrajectorySeedHitCandidate(const FastTrackerRecHit* hit, const TrackerTopology* tTopo)
    : theHit(hit), seedingLayer(TrackingLayer::createFromDetId(hit->geographicalId(), *tTopo)) {}

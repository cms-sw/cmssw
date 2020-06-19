//#define EDM_ML_DEBUG

#include "RecoMTD/DetLayers/interface/MTDDetSector.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>

using namespace std;

MTDDetSector::MTDDetSector(vector<const GeomDet*>::const_iterator first, vector<const GeomDet*>::const_iterator last)
    : GeometricSearchDet(false), theDets(first, last) {
  init();
}

MTDDetSector::MTDDetSector(const vector<const GeomDet*>& vdets) : GeometricSearchDet(false), theDets(vdets) { init(); }

void MTDDetSector::init() {
  //theBinFinder = BinFinderType(basicComponents().front()->position().phi(), basicComponents().size());
}

MTDDetSector::~MTDDetSector() {}

const vector<const GeometricSearchDet*>& MTDDetSector::components() const {
  // FIXME dummy impl.
  edm::LogError("MTDDetLayers") << "temporary dummy implementation of MTDDetSector::components()!!";
  static const vector<const GeometricSearchDet*> result;
  return result;
}

pair<bool, TrajectoryStateOnSurface> MTDDetSector::compatible(const TrajectoryStateOnSurface& ts,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est) const {
  TrajectoryStateOnSurface ms = prop.propagate(ts, specificSurface());

#ifdef EDM_ML_DEBUG
  LogTrace("MTDDetLayers") << "MTDDetSector::compatible, Surface at Z: " << specificSurface().position().z()
                           << " R1: " << specificSurface().innerRadius() << " R2: " << specificSurface().outerRadius()
                           << " TS   at Z,R: " << ts.globalPosition().z() << "," << ts.globalPosition().perp();
  if (ms.isValid()) {
    LogTrace("MTDDetLayers") << " DEST at Z,R: " << ms.globalPosition().z() << "," << ms.globalPosition().perp()
                             << " local Z: " << ms.localPosition().z();
  } else {
    LogTrace("MTDDetLayers") << " DEST: not valid";
  }
#endif

  if (ms.isValid())
    return make_pair(est.estimate(ms, specificSurface()) != 0, ms);
  else
    return make_pair(false, ms);
}

vector<GeometricSearchDet::DetWithState> MTDDetSector::compatibleDets(const TrajectoryStateOnSurface& startingState,
                                                                      const Propagator& prop,
                                                                      const MeasurementEstimator& est) const {
  LogTrace("MTDDetLayers") << "MTDDetSector::compatibleDets, Surface at Z: " << surface().position().z()
                           << " R1: " << specificSurface().innerRadius() << " R2: " << specificSurface().outerRadius()
                           << " TS at Z,R: " << startingState.globalPosition().z() << ","
                           << startingState.globalPosition().perp() << "     DetRing pos." << position();

  vector<DetWithState> result;

  // Propagate and check that the result is within bounds
  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);
  if (!compat.first) {
    LogTrace("MTDDetLayers") << "    MTDDetSector::compatibleDets: not compatible"
                             << "    (should not have been selected!)";
    return result;
  }

  return result;
}

void MTDDetSector::compatibleDetsV(const TrajectoryStateOnSurface&,
                                   const Propagator&,
                                   const MeasurementEstimator&,
                                   std::vector<DetWithState>&) const {
  edm::LogError("MTDDetLayers") << "At the moment not a real implementation";
}

vector<DetGroup> MTDDetSector::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                     const Propagator& prop,
                                                     const MeasurementEstimator& est) const {
  // FIXME should be implemented to allow returning  overlapping chambers
  // as separate groups!
  edm::LogInfo("MTDDetLayers") << "dummy implementation of MTDDetSector::groupedCompatibleDets()";
  vector<DetGroup> result;
  return result;
}

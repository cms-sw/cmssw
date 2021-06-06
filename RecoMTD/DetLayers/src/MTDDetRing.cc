//#define EDM_ML_DEBUG

/** \file
 *
 *  \author L. Gray - FNAL
 */

#include "RecoMTD/DetLayers/interface/MTDDetRing.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>

using namespace std;

MTDDetRing::MTDDetRing(vector<const GeomDet*>::const_iterator first, vector<const GeomDet*>::const_iterator last)
    : ForwardDetRingOneZ(first, last) {
  init();
}

MTDDetRing::MTDDetRing(const vector<const GeomDet*>& vdets) : ForwardDetRingOneZ(vdets) { init(); }

void MTDDetRing::init() {
  theBinFinder = BinFinderType(basicComponents().front()->position().phi(), basicComponents().size());
}

MTDDetRing::~MTDDetRing() {}

const vector<const GeometricSearchDet*>& MTDDetRing::components() const {
  // FIXME dummy impl.
  edm::LogError("MTDDetLayers") << "temporary dummy implementation of MTDDetRing::components()!!";
  static const vector<const GeometricSearchDet*> result;
  return result;
}

pair<bool, TrajectoryStateOnSurface> MTDDetRing::compatible(const TrajectoryStateOnSurface& ts,
                                                            const Propagator& prop,
                                                            const MeasurementEstimator& est) const {
  TrajectoryStateOnSurface ms = prop.propagate(ts, specificSurface());

#ifdef EDM_ML_DEBUG
  LogTrace("MTDDetLayers") << "MTDDetRing::compatible, Surface at Z: " << specificSurface().position().z()
                           << " R1: " << specificSurface().innerRadius() << " R2: " << specificSurface().outerRadius()
                           << " TS   at Z,R: " << ts.globalPosition().z() << "," << ts.globalPosition().perp();
  if (ms.isValid()) {
    LogTrace("MTDDetLayers") << " DEST at Z,R: " << ms.globalPosition().z() << "," << ms.globalPosition().perp()
                             << " local Z: " << ms.localPosition().z();
  } else {
    LogTrace("MTDDetLayers") << " DEST: not valid";
  }
#endif

  return make_pair(ms.isValid() and est.estimate(ms, specificSurface()) != 0, ms);
}

vector<GeometricSearchDet::DetWithState> MTDDetRing::compatibleDets(const TrajectoryStateOnSurface& startingState,
                                                                    const Propagator& prop,
                                                                    const MeasurementEstimator& est) const {
  LogTrace("MTDDetLayers") << "MTDDetRing::compatibleDets, Surface at Z: " << surface().position().z()
                           << " R1: " << specificSurface().innerRadius() << " R2: " << specificSurface().outerRadius()
                           << " TS at Z,R: " << startingState.globalPosition().z() << ","
                           << startingState.globalPosition().perp() << "     DetRing pos." << position();

  vector<DetWithState> result;

  // Propagate and check that the result is within bounds
  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);
  if (!compat.first) {
    LogTrace("MTDDetLayers") << "    MTDDetRing::compatibleDets: not compatible"
                             << "    (should not have been selected!)";
    return result;
  }

  // Find the most probable destination component
  TrajectoryStateOnSurface& tsos = compat.second;
  GlobalPoint startPos = tsos.globalPosition();
  int closest = theBinFinder.binIndex(startPos.phi());
  const vector<const GeomDet*> dets = basicComponents();
  LogTrace("MTDDetLayers") << "     MTDDetRing::compatibleDets, closest det: " << closest
                           << " Phi: " << dets[closest]->surface().position().phi() << " impactPhi " << startPos.phi();

  // Add this detector, if it is compatible
  // NOTE: add performs a null propagation
  add(closest, result, tsos, prop, est);

#ifdef EDM_ML_DEBUG
  int nclosest = result.size();
  int nnextdet = 0;  // MDEBUG counters

  // Try the neighbors on each side until no more compatible.
  float dphi = 0;
  if (!result.empty()) {  // If closest is not compatible the next cannot be either
    float nSigmas = 3.;
    if (result.back().second.hasError()) {
      dphi = nSigmas * atan(sqrt(result.back().second.localError().positionError().xx()) /
                            result.back().second.globalPosition().perp());
    }
  } else {
    LogTrace("MTDDetLayers") << "     MTDDetRing::compatibleDets, closest not compatible!";
    //FIXME:  if closest is not compatible the next cannot be either
  }
#endif

  for (int idet = closest + 1; idet < closest + int(dets.size()) / 4 + 1; idet++) {
    // FIXME: should use dphi to decide if det must be queried.
    // Right now query until not compatible.
    int idetp = theBinFinder.binIndex(idet);
    {
#ifdef EDM_ML_DEBUG
      LogTrace("MTDDetLayers") << "     next det:" << idetp << " at Z: " << dets[idetp]->position().z()
                               << " phi: " << dets[idetp]->position().phi() << " FTS phi " << startPos.phi()
                               << " max dphi " << dphi;
      nnextdet++;
#endif
      if (!add(idetp, result, tsos, prop, est))
        break;
    }
  }

  for (int idet = closest - 1; idet > closest - int(dets.size()) / 4 - 1; idet--) {
    // FIXME: should use dphi to decide if det must be queried.
    // Right now query until not compatible.
    int idetp = theBinFinder.binIndex(idet);
    {
#ifdef EDM_ML_DEBUG
      LogTrace("MTDDetLayers") << "     previous det:" << idetp << " " << idet << " " << closest - dets.size() / 4 - 1
                               << " at Z: " << dets[idetp]->position().z() << " phi: " << dets[idetp]->position().phi()
                               << " FTS phi " << startPos.phi() << " max dphi" << dphi;
      nnextdet++;
#endif
      if (!add(idetp, result, tsos, prop, est))
        break;
    }
  }

#ifdef EDM_ML_DEBUG
  LogTrace("MTDDetLayers") << "     MTDDetRing::compatibleDets, size: " << result.size() << " on closest: " << nclosest
                           << " # checked dets: " << nnextdet + 1;

  if (result.empty()) {
    LogTrace("MTDDetLayers") << "   ***Ring not compatible,should have been discarded before!!!";
  }
#endif

  return result;
}

vector<DetGroup> MTDDetRing::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                   const Propagator& prop,
                                                   const MeasurementEstimator& est) const {
  // FIXME should be implemented to allow returning  overlapping chambers
  // as separate groups!
  edm::LogError("MTDDetLayers") << "dummy implementation of MTDDetRing::groupedCompatibleDets()";
  vector<DetGroup> result;
  return result;
}

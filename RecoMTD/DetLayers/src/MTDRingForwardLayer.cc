//#define EDM_ML_DEBUG

/** \file
 *
 *  \author L. Gray - FNAL
 */

#include <RecoMTD/DetLayers/interface/MTDRingForwardLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetRing.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <DataFormats/GeometrySurface/interface/SimpleDiskBounds.h>
#include <TrackingTools/GeomPropagators/interface/Propagator.h>
#include <TrackingTools/DetLayers/interface/MeasurementEstimator.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "TrackingTools/DetLayers/interface/RBorderFinder.h"
#include "TrackingTools/DetLayers/interface/GeneralBinFinderInR.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

MTDRingForwardLayer::MTDRingForwardLayer(const vector<const ForwardDetRing*>& rings)
    : RingedForwardLayer(false),
      theRings(rings),
      theComponents(theRings.begin(), theRings.end()),
      theBinFinder(nullptr),
      isOverlapping(false) {
  // Initial values for R and Z bounds
  float theRmin = rings.front()->basicComponents().front()->position().perp();
  float theRmax = theRmin;
  float theZmin = rings.front()->position().z();
  float theZmax = theZmin;

  // Cache chamber pointers (the basic components_)
  // and find extension in R and Z
  for (const auto& it : rings) {
    vector<const GeomDet*> tmp2 = it->basicComponents();
    theBasicComps.insert(theBasicComps.end(), tmp2.begin(), tmp2.end());

    theRmin = min(theRmin, it->specificSurface().innerRadius());
    theRmax = max(theRmax, it->specificSurface().outerRadius());
    float halfThick = it->surface().bounds().thickness() / 2.;
    float zCenter = it->surface().position().z();
    theZmin = min(theZmin, zCenter - halfThick);
    theZmax = max(theZmax, zCenter + halfThick);
  }

  RBorderFinder bf(theRings);
  isOverlapping = bf.isROverlapping();
  theBinFinder = new GeneralBinFinderInR<double>(bf);

  // Build surface

  float zPos = (theZmax + theZmin) / 2.;
  PositionType pos(0., 0., zPos);
  RotationType rot;

  setSurface(new BoundDisk(pos, rot, new SimpleDiskBounds(theRmin, theRmax, theZmin - zPos, theZmax - zPos)));

  LogTrace("MTDDetLayers") << "Constructing MTDRingForwardLayer: " << basicComponents().size() << " Dets "
                           << theRings.size() << " Rings "
                           << " Z: " << specificSurface().position().z() << " R1: " << specificSurface().innerRadius()
                           << " R2: " << specificSurface().outerRadius() << " Per.: " << bf.isRPeriodic()
                           << " Overl.: " << bf.isROverlapping();
}

MTDRingForwardLayer::~MTDRingForwardLayer() {
  delete theBinFinder;
  for (auto& i : theRings) {
    delete i;
  }
}

vector<GeometricSearchDet::DetWithState> MTDRingForwardLayer::compatibleDets(
    const TrajectoryStateOnSurface& startingState, const Propagator& prop, const MeasurementEstimator& est) const {
  vector<DetWithState> result;

  LogTrace("MTDDetLayers") << "MTDRingForwardLayer::compatibleDets,"
                           << " R1 " << specificSurface().innerRadius() << " R2: " << specificSurface().outerRadius()
                           << " FTS at R: " << startingState.globalPosition().perp();

  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);

  if (!compat.first) {
    LogTrace("MTDDetLayers") << "     MTDRingForwardLayer::compatibleDets: not compatible"
                             << " (should not have been selected!)";
    return result;
  }

  TrajectoryStateOnSurface& tsos = compat.second;

  int closest = theBinFinder->binIndex(tsos.globalPosition().perp());
  const ForwardDetRing* closestRing = theRings[closest];

  // Check the closest ring

  LogTrace("MTDDetLayers") << "     MTDRingForwardLayer::fastCompatibleDets, closestRing: " << closest << " R1 "
                           << closestRing->specificSurface().innerRadius()
                           << " R2: " << closestRing->specificSurface().outerRadius()
                           << " FTS R: " << tsos.globalPosition().perp();
  if (tsos.hasError()) {
    LogTrace("MTDDetLayers") << " sR: " << sqrt(tsos.localError().positionError().yy())
                             << " sX: " << sqrt(tsos.localError().positionError().xx());
  }

  result = closestRing->compatibleDets(tsos, prop, est);

#ifdef EDM_ML_DEBUG
  int nclosest = result.size();
  int nnextdet = 0;  // MDEBUG counters
#endif

  //FIXME: if closest is not compatible next cannot be either?

  // Use state on layer surface. Note that local coordinates and errors
  // are the same on the layer and on all rings surfaces, since
  // all BoundDisks are centered in 0,0 and have the same rotation.
  // CAVEAT: if the rings are not at the same Z, the local position and error
  // will be "Z-projected" to the rings. This is a fairly good approximation.
  // However in this case additional propagation will be done when calling
  // compatibleDets.
  GlobalPoint startPos = tsos.globalPosition();
  LocalPoint nextPos(surface().toLocal(startPos));

  for (unsigned int idet = closest + 1; idet < theRings.size(); idet++) {
    bool inside = false;
    if (tsos.hasError()) {
      inside = theRings[idet]->specificSurface().bounds().inside(nextPos, tsos.localError().positionError());
    } else {
      inside = theRings[idet]->specificSurface().bounds().inside(nextPos);
    }
    if (inside) {
#ifdef EDM_ML_DEBUG
      LogTrace("MTDDetLayers") << "     MTDRingForwardLayer::fastCompatibleDets:NextRing" << idet << " R1 "
                               << theRings[idet]->specificSurface().innerRadius()
                               << " R2: " << theRings[idet]->specificSurface().outerRadius() << " FTS R "
                               << nextPos.perp();
      nnextdet++;
#endif
      vector<DetWithState> nextRodDets = theRings[idet]->compatibleDets(tsos, prop, est);
      if (!nextRodDets.empty()) {
        result.insert(result.end(), nextRodDets.begin(), nextRodDets.end());
      } else {
        break;
      }
    }
  }

  for (int idet = closest - 1; idet >= 0; idet--) {
    bool inside = false;
    if (tsos.hasError()) {
      inside = theRings[idet]->specificSurface().bounds().inside(nextPos, tsos.localError().positionError());
    } else {
      inside = theRings[idet]->specificSurface().bounds().inside(nextPos);
    }
    if (inside) {
#ifdef EDM_ML_DEBUG
      LogTrace("MTDDetLayers") << "     MTDRingForwardLayer::fastCompatibleDets:PreviousRing:" << idet << " R1 "
                               << theRings[idet]->specificSurface().innerRadius()
                               << " R2: " << theRings[idet]->specificSurface().outerRadius() << " FTS R "
                               << nextPos.perp();
      nnextdet++;
#endif
      vector<DetWithState> nextRodDets = theRings[idet]->compatibleDets(tsos, prop, est);
      if (!nextRodDets.empty()) {
        result.insert(result.end(), nextRodDets.begin(), nextRodDets.end());
      } else {
        break;
      }
    }
  }

#ifdef EDM_ML_DEBUG
  LogTrace("MTDDetLayers") << "     MTDRingForwardLayer::fastCompatibleDets: found: " << result.size()
                           << " on closest: " << nclosest << " # checked rings: " << 1 + nnextdet;
#endif

  return result;
}

vector<DetGroup> MTDRingForwardLayer::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                            const Propagator& prop,
                                                            const MeasurementEstimator& est) const {
  // FIXME should return only 1 group
  edm::LogError("MTDDetLayers") << "dummy implementation of MTDRingForwardLayer::groupedCompatibleDets()";
  return vector<DetGroup>();
}

GeomDetEnumerators::SubDetector MTDRingForwardLayer::subDetector() const {
  return theBasicComps.front()->subDetector();
}

const vector<const GeometricSearchDet*>& MTDRingForwardLayer::components() const { return theComponents; }

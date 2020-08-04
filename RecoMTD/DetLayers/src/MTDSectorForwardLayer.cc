#define EDM_ML_DEBUG

#include <RecoMTD/DetLayers/interface/MTDSectorForwardLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetSector.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <DataFormats/GeometrySurface/interface/DiskSectorBounds.h>
#include <TrackingTools/GeomPropagators/interface/Propagator.h>
#include <TrackingTools/DetLayers/interface/MeasurementEstimator.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

MTDSectorForwardLayer::MTDSectorForwardLayer(const vector<const MTDDetSector*>& sectors)
    : ForwardDetLayer(false), theSectors(sectors), theComponents(theSectors.begin(), theSectors.end()) {
  // Initial values for R, Z and Phi bounds
  float theRmin = sectors.front()->basicComponents().front()->position().perp();
  float theRmax = theRmin;
  float theZmin = sectors.front()->position().z();
  float theZmax = theZmin;

  // Cache chamber pointers (the basic components_)
  // and find extension in R and Z
  for (vector<const MTDDetSector*>::const_iterator it = sectors.begin(); it != sectors.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(), tmp2.begin(), tmp2.end());

    theRmin = min(theRmin, (*it)->specificSurface().innerRadius());
    theRmax = max(theRmax, (*it)->specificSurface().outerRadius());
    float halfThick = (*it)->surface().bounds().thickness() / 2.;
    float zCenter = (*it)->surface().position().z();
    theZmin = min(theZmin, zCenter - halfThick);
    theZmax = max(theZmax, zCenter + halfThick);
  }

  // Build surface

  float zPos = (theZmax + theZmin) / 2.;
  PositionType pos(0., 0., zPos);
  RotationType rot;

  setSurface(new BoundDisk(pos, rot, new SimpleDiskBounds(theRmin, theRmax, theZmin - zPos, theZmax - zPos)));

  LogTrace("MTDDetLayers") << "Constructing MTDSectorForwardLayer: " << basicComponents().size() << " Dets "
                           << theSectors.size() << " Sectors "
                           << " Z: " << specificSurface().position().z() << " R1: " << specificSurface().innerRadius()
                           << " R2: " << specificSurface().outerRadius();
}

MTDSectorForwardLayer::~MTDSectorForwardLayer() {
  for (vector<const MTDDetSector*>::iterator i = theSectors.begin(); i < theSectors.end(); i++) {
    delete *i;
  }
}

vector<GeometricSearchDet::DetWithState> MTDSectorForwardLayer::compatibleDets(
    const TrajectoryStateOnSurface& startingState, const Propagator& prop, const MeasurementEstimator& est) const {
  vector<DetWithState> result;

  LogTrace("MTDDetLayers") << "MTDSectorForwardLayer::compatibleDets,"
                           << " R1 " << specificSurface().innerRadius() << " R2: " << specificSurface().outerRadius()
                           << " FTS at R: " << startingState.globalPosition().perp();

  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);

  if (!compat.first) {
    LogTrace("MTDDetLayers") << "     MTDSectorForwardLayer::compatibleDets: not compatible"
                             << " (should not have been selected!)";
    return result;
  }

  TrajectoryStateOnSurface& tsos = compat.second;

  // as there are either two or four sectors only, avoid complex logic and just loop on all of them

  // Use state on layer surface. Note that local coordinates and errors
  // are the same on the layer and on all sectors surfaces, since
  // all BoundDisks are centered in 0,0 and have the same rotation.
  // CAVEAT: if the sectors are not at the same Z, the local position and error
  // will be "Z-projected" to the sectors. This is a fairly good approximation.
  // However in this case additional propagation will be done when calling
  // compatibleDets.
  GlobalPoint startPos = tsos.globalPosition();

  for (unsigned int isect = 0; isect < theSectors.size(); isect++) {
    LocalPoint nextPos(theSectors[isect]->specificSurface().toLocal(startPos));
    LogDebug("MTDDetLayers") << "Global point = " << startPos << " local point = " << nextPos;
    bool inside = false;
    if (tsos.hasError()) {
      inside = theSectors[isect]->specificSurface().bounds().inside(nextPos, tsos.localError().positionError(), 1.);
    } else {
      inside = theSectors[isect]->specificSurface().bounds().inside(nextPos);
    }
    if (inside) {
#ifdef EDM_ML_DEBUG
      LogTrace("MTDDetLayers") << "     MTDSectorForwardLayer::fastCompatibleDets:NextSector " << isect << " R1 "
                               << theSectors[isect]->specificSurface().innerRadius()
                               << " R2: " << theSectors[isect]->specificSurface().outerRadius() << " PhiMin: "
                               << theSectors[isect]->specificSurface().position().phi() -
                                      theSectors[isect]->specificSurface().phiHalfExtension()
                               << " PhiMax: "
                               << theSectors[isect]->specificSurface().position().phi() +
                                      theSectors[isect]->specificSurface().phiHalfExtension()
                               << " FTS R,phi: " << tsos.globalPosition().perp() << "," << tsos.globalPosition().phi();
      if (tsos.hasError()) {
        LogTrace("MTDDetLayers") << " sR: " << sqrt(tsos.localError().positionError().yy())
                                 << " sX: " << sqrt(tsos.localError().positionError().xx());
      }
#endif
      vector<DetWithState> nextRodDets = theSectors[isect]->compatibleDets(tsos, prop, est);
      if (!nextRodDets.empty()) {
        result.insert(result.end(), nextRodDets.begin(), nextRodDets.end());
      } else {
        break;
      }
    }
  }

#ifdef EDM_ML_DEBUG
  LogTrace("MTDDetLayers") << "     MTDSectorForwardLayer::fastCompatibleDets: found: " << result.size();
#endif

  return result;
}

vector<DetGroup> MTDSectorForwardLayer::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est) const {
  // FIXME should return only 1 group
  edm::LogInfo("MTDDetLayers") << "dummy implementation of MTDSectorForwardLayer::groupedCompatibleDets()";
  return vector<DetGroup>();
}

GeomDetEnumerators::SubDetector MTDSectorForwardLayer::subDetector() const {
  return theBasicComps.front()->subDetector();
}

const vector<const GeometricSearchDet*>& MTDSectorForwardLayer::components() const { return theComponents; }

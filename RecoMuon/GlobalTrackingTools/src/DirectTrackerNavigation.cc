/** 
 *  Class: DirectTrackerNavigation
 *
 *
 *
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"

//---------------
// C++ Headers --
//---------------

#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

using namespace std;

//
// constructor
//
DirectTrackerNavigation::DirectTrackerNavigation(const edm::ESHandle<GeometricSearchTracker>& tkLayout, bool outOnly)
    : theGeometricSearchTracker(tkLayout), theOutLayerOnlyFlag(outOnly), theEpsilon(-0.01) {}

//
// return compatible layers for a given trajectory state
//
vector<const DetLayer*> DirectTrackerNavigation::compatibleLayers(const FreeTrajectoryState& fts,
                                                                  PropagationDirection dir) const {
  bool inOut = outward(fts);
  double eta0 = fts.position().eta();

  vector<const DetLayer*> output;

  // check eta of DetLayers for compatibility

  if (inOut) {
    if (!theOutLayerOnlyFlag) {
      inOutPx(fts, output);

      if (eta0 > 1.55)
        inOutFPx(fts, output);
      else if (eta0 < -1.55)
        inOutBPx(fts, output);

      if (fabs(eta0) < 1.67)
        inOutTIB(fts, output);

      if (eta0 > 1.17)
        inOutFTID(fts, output);
      else if (eta0 < -1.17)
        inOutBTID(fts, output);
    }

    if (fabs(eta0) < 1.35)
      inOutTOB(fts, output);

    if (eta0 > 0.97)
      inOutFTEC(fts, output);
    else if (eta0 < -0.97)
      inOutBTEC(fts, output);

  } else {
    LogTrace("Muon|RecoMuon|DirectionTrackerNavigation") << "No implementation for inward state at this moment. ";
  }

  if (dir == oppositeToMomentum)
    std::reverse(output.begin(), output.end());

  return output;
}

//
//
//
void DirectTrackerNavigation::inOutPx(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->pixelBarrelLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutTIB(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->tibLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutTOB(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->tobLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutFPx(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->posPixelForwardLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutFTID(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->posTidLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutFTEC(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->posTecLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutBPx(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->negPixelForwardLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutBTID(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->negTidLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
void DirectTrackerNavigation::inOutBTEC(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  for (const auto i : theGeometricSearchTracker->negTecLayers()) {
    if (checkCompatible(fts, i))
      output.push_back(i);
  }
}

//
//
//
bool DirectTrackerNavigation::checkCompatible(const FreeTrajectoryState& fts, const BarrelDetLayer* dl) const {
  float eta0 = fts.position().eta();

  const BoundCylinder& bc = dl->specificSurface();
  float radius = bc.radius();
  float length = bc.bounds().length() / 2.;

  float eta = calculateEta(radius, length);

  return (fabs(eta0) <= (fabs(eta) + theEpsilon));
}

//
//
//
bool DirectTrackerNavigation::checkCompatible(const FreeTrajectoryState& fts, const ForwardDetLayer* dl) const {
  float eta0 = fts.position().eta();

  const BoundDisk& bd = dl->specificSurface();

  float outRadius = bd.outerRadius();
  float inRadius = bd.innerRadius();
  float z = bd.position().z();

  float etaOut = calculateEta(outRadius, z);
  float etaIn = calculateEta(inRadius, z);

  if (eta0 > 0)
    return (eta0 > (etaOut - theEpsilon) && eta0 < (etaIn + theEpsilon));
  else
    return (eta0 < (etaOut + theEpsilon) && eta0 > (etaIn - theEpsilon));
}

//
//
//
bool DirectTrackerNavigation::outward(const FreeTrajectoryState& fts) const {
  return (fts.position().basicVector().dot(fts.momentum().basicVector()) > 0);
}

//
// calculate pseudorapidity from r and z
//
float DirectTrackerNavigation::calculateEta(float r, float z) const {
  if (z > 0)
    return -log((tan(atan(r / z) / 2.)));
  return log(-(tan(atan(r / z) / 2.)));
}

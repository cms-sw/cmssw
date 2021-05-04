//#define EDM_ML_DEBUG

#include <RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetSector.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <DataFormats/GeometrySurface/interface/SimpleDiskBounds.h>
#include <TrackingTools/GeomPropagators/interface/Propagator.h>
#include <TrackingTools/DetLayers/interface/MeasurementEstimator.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

MTDSectorForwardDoubleLayer::MTDSectorForwardDoubleLayer(const vector<const MTDDetSector*>& frontSectors,
                                                         const vector<const MTDDetSector*>& backSectors)
    : ForwardDetLayer(true),
      theFrontLayer(frontSectors),
      theBackLayer(backSectors),
      theSectors(frontSectors),  // add back later
      theComponents(),
      theBasicComponents() {
  theSectors.insert(theSectors.end(), backSectors.begin(), backSectors.end());
  theComponents = std::vector<const GeometricSearchDet*>(theSectors.begin(), theSectors.end());

  // Cache chamber pointers (the basic components_)
  // and find extension in R and Z
  for (const auto& isect : theSectors) {
    vector<const GeomDet*> tmp2 = isect->basicComponents();
    theBasicComponents.insert(theBasicComponents.end(), tmp2.begin(), tmp2.end());
  }

  setSurface(computeSurface());

  LogTrace("MTDDetLayers") << "Constructing MTDSectorForwardDoubleLayer: " << std::fixed << std::setw(14)
                           << basicComponents().size() << " Dets " << std::setw(14) << theSectors.size() << " Sectors "
                           << " Z: " << std::setw(14) << specificSurface().position().z() << " R1: " << std::setw(14)
                           << specificSurface().innerRadius() << " R2: " << std::setw(14)
                           << specificSurface().outerRadius();

  selfTest();
}

BoundDisk* MTDSectorForwardDoubleLayer::computeSurface() {
  const BoundDisk& frontDisk = theFrontLayer.specificSurface();
  const BoundDisk& backDisk = theBackLayer.specificSurface();

  float rmin = min(frontDisk.innerRadius(), backDisk.innerRadius());
  float rmax = max(frontDisk.outerRadius(), backDisk.outerRadius());
  float zmin = frontDisk.position().z();
  float halfThickness = frontDisk.bounds().thickness() / 2.;
  zmin = (zmin > 0) ? zmin - halfThickness : zmin + halfThickness;
  float zmax = backDisk.position().z();
  halfThickness = backDisk.bounds().thickness() / 2.;
  zmax = (zmax > 0) ? zmax + halfThickness : zmax - halfThickness;
  float zPos = (zmax + zmin) / 2.;
  PositionType pos(0., 0., zPos);
  RotationType rot;

  return new BoundDisk(pos, rot, new SimpleDiskBounds(rmin, rmax, zmin - zPos, zmax - zPos));
}

bool MTDSectorForwardDoubleLayer::isInsideOut(const TrajectoryStateOnSurface& tsos) const {
  return tsos.globalPosition().basicVector().dot(tsos.globalMomentum().basicVector()) > 0;
}

std::pair<bool, TrajectoryStateOnSurface> MTDSectorForwardDoubleLayer::compatible(
    const TrajectoryStateOnSurface& startingState, const Propagator& prop, const MeasurementEstimator& est) const {
  // mostly copied from ForwardDetLayer, except propagates to closest surface,
  // not to center

  bool insideOut = isInsideOut(startingState);
  const MTDSectorForwardLayer& closerLayer = (insideOut) ? theFrontLayer : theBackLayer;
  LogTrace("MTDDetLayers") << "MTDSectorForwardDoubleLayer::compatible is assuming inside-out direction: " << insideOut;

  TrajectoryStateOnSurface myState = prop.propagate(startingState, closerLayer.specificSurface());
  if (!myState.isValid())
    return make_pair(false, myState);

  // take into account the thickness of the layer
  float deltaR = surface().bounds().thickness() / 2. * std::abs(tan(myState.localDirection().theta()));

  // take into account the error on the predicted state
  constexpr float nSigma = 3.;
  if (myState.hasError()) {
    LocalError err = myState.localError().positionError();
    // ignore correlation for the moment...
    deltaR += nSigma * sqrt(err.xx() + err.yy());
  }

  float zPos = (zmax() + zmin()) / 2.;
  SimpleDiskBounds tmp(rmin() - deltaR, rmax() + deltaR, zmin() - zPos, zmax() - zPos);

  return make_pair(tmp.inside(myState.localPosition()), myState);
}

vector<GeometricSearchDet::DetWithState> MTDSectorForwardDoubleLayer::compatibleDets(
    const TrajectoryStateOnSurface& startingState, const Propagator& prop, const MeasurementEstimator& est) const {
  vector<DetWithState> result;
  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);

  if (!compat.first) {
    LogTrace("MTDDetLayers") << "     MTDSectorForwardDoubleLayer::compatibleDets: not compatible"
                             << " (should not have been selected!)";
    return result;
  }

  TrajectoryStateOnSurface& tsos = compat.second;

  // standard implementation of compatibleDets() for class which have
  // groupedCompatibleDets implemented.
  // This code should be moved in a common place intead of being
  // copied many times.
  vector<DetGroup> vectorGroups(groupedCompatibleDets(tsos, prop, est));
  for (const auto& thisDG : vectorGroups) {
    for (const auto& thisDGE : thisDG) {
      result.emplace_back(DetWithState(thisDGE.det(), thisDGE.trajectoryState()));
    }
  }

  return result;
}

vector<DetGroup> MTDSectorForwardDoubleLayer::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                                    const Propagator& prop,
                                                                    const MeasurementEstimator& est) const {
  vector<GeometricSearchDet::DetWithState> detWithStates1, detWithStates2;

  LogTrace("MTDDetLayers") << "groupedCompatibleDets are currently given always in inside-out order";
  // this should be fixed either in RecoMTD/MeasurementDet/MTDDetLayerMeasurements or
  // RecoMTD/DetLayers/MTDSectorForwardDoubleLayer

  detWithStates1 = theFrontLayer.compatibleDets(startingState, prop, est);
  detWithStates2 = theBackLayer.compatibleDets(startingState, prop, est);

  vector<DetGroup> result;
  if (!detWithStates1.empty())
    result.push_back(DetGroup(detWithStates1));
  if (!detWithStates2.empty())
    result.push_back(DetGroup(detWithStates2));
  LogTrace("MTDDetLayers") << "DoubleLayer Compatible dets: " << result.size();
  return result;
}

bool MTDSectorForwardDoubleLayer::isCrack(const GlobalPoint& gp) const {
  bool result = false;
  LogTrace("MTDDetLayers")
      << "MTDSectorForwardDoubleLayer::isCrack kept only for backward compatibility, no real implementation";
  return result;
}

void MTDSectorForwardDoubleLayer::selfTest() const {
  const std::vector<const GeomDet*>& frontDets = theFrontLayer.basicComponents();
  const std::vector<const GeomDet*>& backDets = theBackLayer.basicComponents();

  // test that each front z is less than each back z
  float frontz(0.);
  float backz(1e3f);
  for (const auto& thisFront : frontDets) {
    float tmpz(std::abs(thisFront->surface().position().z()));
    if (tmpz > frontz) {
      frontz = tmpz;
    }
  }
  for (const auto& thisBack : backDets) {
    float tmpz(std::abs(thisBack->surface().position().z()));
    if (tmpz < backz) {
      backz = tmpz;
    }
  }
  assert(frontz < backz);
}

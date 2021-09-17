/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/GeneralBinFinderInPhi.h"
#include "TrackingTools/DetLayers/interface/PhiBorderFinder.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Utilities/General/interface/precomputed_value_sort.h"
#include "DataFormats/GeometrySurface/interface/GeometricSorting.h"

#include <algorithm>
#include <iostream>

using namespace std;

MuRodBarrelLayer::MuRodBarrelLayer(vector<const DetRod*>& rods)
    : RodBarrelLayer(false), theRods(rods), theBinFinder(nullptr), isOverlapping(false) {
  // Sort rods in phi
  precomputed_value_sort(theRods.begin(), theRods.end(), geomsort::ExtractPhi<DetRod, float>());

  theComponents.reserve(theRods.size());
  std::copy(theRods.begin(), theRods.end(), back_inserter(theComponents));

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuRodBarrelLayer";

  // Cache chamber pointers (the basic components_)
  for (vector<const DetRod*>::const_iterator it = rods.begin(); it != rods.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(), tmp2.begin(), tmp2.end());
  }

  // Initialize the binfinder
  PhiBorderFinder bf(theRods);
  isOverlapping = bf.isPhiOverlapping();

  if (bf.isPhiPeriodic()) {
    theBinFinder = new PeriodicBinFinderInPhi<double>(theRods.front()->position().phi(), theRods.size());
  } else {
    theBinFinder = new GeneralBinFinderInPhi<double>(bf);
  }

  // Compute the layer's surface and bounds (from the components())
  BarrelDetLayer::initialize();

  LogTrace(metname) << "Constructing MuRodBarrelLayer: " << basicComponents().size() << " Dets " << theRods.size()
                    << " Rods "
                    << " R: " << specificSurface().radius() << " Per.: " << bf.isPhiPeriodic()
                    << " Overl.: " << isOverlapping;
}

MuRodBarrelLayer::~MuRodBarrelLayer() {
  delete theBinFinder;
  for (vector<const DetRod*>::iterator i = theRods.begin(); i < theRods.end(); i++) {
    delete *i;
  }
}

vector<GeometricSearchDet::DetWithState> MuRodBarrelLayer::compatibleDets(const TrajectoryStateOnSurface& startingState,
                                                                          const Propagator& prop,
                                                                          const MeasurementEstimator& est) const {
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuRodBarrelLayer";
  vector<DetWithState> result;

  LogTrace(metname) << "MuRodBarrelLayer::compatibleDets, Cyl R: " << specificSurface().radius()
                    << " TSOS at R= " << startingState.globalPosition().perp()
                    << " phi= " << startingState.globalPosition().phi();

  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);
  if (!compat.first) {
    LogTrace(metname) << "     MuRodBarrelLayer::compatibleDets: not compatible"
                      << " (should not have been selected!)";
    return vector<DetWithState>();
  }

  TrajectoryStateOnSurface& tsos = compat.second;

  LogTrace(metname) << "     MuRodBarrelLayer::compatibleDets, reached layer at: " << tsos.globalPosition()
                    << " R = " << tsos.globalPosition().perp() << " phi = " << tsos.globalPosition().phi();

  int closest = theBinFinder->binIndex(tsos.globalPosition().phi());
  const DetRod* closestRod = theRods[closest];

  // Check the closest rod
  LogTrace(metname) << "     MuRodBarrelLayer::compatibleDets, closestRod: " << closest
                    << " phi : " << closestRod->surface().position().phi()
                    << " FTS phi: " << tsos.globalPosition().phi();

  result = closestRod->compatibleDets(tsos, prop, est);

  int nclosest = result.size();  // Debug counter

  bool checknext = false;
  double dist;

  if (!result.empty()) {
    // Check if the track go outside closest rod, then look for closest.
    TrajectoryStateOnSurface& predictedState = result.front().second;
    float xErr = xError(predictedState, est);
    float halfWid = closestRod->surface().bounds().width() / 2.;
    dist = predictedState.localPosition().x();

    // If the layer is overlapping, additionally reduce halfWid by 10%
    // to account for overlap.
    // FIXME: should we account for the real amount of overlap?
    if (isOverlapping)
      halfWid *= 0.9;

    if (fabs(dist) + xErr > halfWid) {
      checknext = true;
    }
  } else {  // Rod is not compatible
    //FIXME: Usually next cannot be either. Implement proper logic.
    // (in general at least one rod should be when this method is called by
    // compatibleDets() which calls compatible())
    checknext = true;

    // Look for the next-to closest in phi.
    // Note Geom::Phi, subtraction is pi-border-safe
    if (tsos.globalPosition().phi() - closestRod->surface().position().phi() > 0.) {
      dist = -1.;
    } else {
      dist = +1.;
    }

    LogTrace(metname) << "     MuRodBarrelLayer::fastCompatibleDets, none on closest rod!";
  }

  if (checknext) {
    int next;
    if (dist < 0.)
      next = closest + 1;
    else
      next = closest - 1;

    next = theBinFinder->binIndex(next);  // Bin Periodicity
    const DetRod* nextRod = theRods[next];

    LogTrace(metname) << "     MuRodBarrelLayer::fastCompatibleDets, next-to closest"
                      << " rod: " << next << " dist " << dist << " phi : " << nextRod->surface().position().phi()
                      << " FTS phi: " << tsos.globalPosition().phi();

    vector<DetWithState> nextRodDets = nextRod->compatibleDets(tsos, prop, est);
    result.insert(result.end(), nextRodDets.begin(), nextRodDets.end());
  }

  LogTrace(metname) << "     MuRodBarrelLayer::fastCompatibleDets: found: " << result.size()
                    << " on closest: " << nclosest << " # checked rods: " << 1 + int(checknext);

  return result;
}

vector<DetGroup> MuRodBarrelLayer::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                         const Propagator& prop,
                                                         const MeasurementEstimator& est) const {
  // FIXME should return only 1 group
  cout << "dummy implementation of MuRodBarrelLayer::groupedCompatibleDets()" << endl;
  return vector<DetGroup>();
}

GeomDetEnumerators::SubDetector MuRodBarrelLayer::subDetector() const { return theBasicComps.front()->subDetector(); }

const vector<const GeometricSearchDet*>& MuRodBarrelLayer::components() const { return theComponents; }

float MuRodBarrelLayer::xError(const TrajectoryStateOnSurface& tsos, const MeasurementEstimator& est) const {
  const float nSigmas = 3.f;
  if (tsos.hasError()) {
    return nSigmas * sqrt(tsos.localError().positionError().xx());
  } else
    return nSigmas * 0.5;
}

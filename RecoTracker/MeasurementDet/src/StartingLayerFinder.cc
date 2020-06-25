#include "RecoTracker/MeasurementDet/interface/StartingLayerFinder.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <utility>

using namespace std;

namespace {

  typedef std::pair<float, float> Range;

  inline bool rangesIntersect(const Range& a, const Range& b) {
    if (a.first > b.second || b.first > a.second)
      return false;
    else
      return true;
  }
};  // namespace

vector<const DetLayer*> StartingLayerFinder::operator()(const FreeTrajectoryState& aFts, float dr, float dz) const {
  vector<const DetLayer*> mylayers;
  mylayers.reserve(3);

  FreeTrajectoryState fastFts(aFts.parameters());

  //barrel pixel
  TrajectoryStateOnSurface pTsos = thePropagator.propagate(fastFts, firstPixelBarrelLayer()->surface());

  if (pTsos.isValid()) {
    Range barrZRange(
        firstPixelBarrelLayer()->position().z() - 0.5 * (firstPixelBarrelLayer()->surface().bounds().length()),
        firstPixelBarrelLayer()->position().z() + 0.5 * (firstPixelBarrelLayer()->surface().bounds().length()));
    Range trajZRange(pTsos.globalPosition().z() - dz, pTsos.globalPosition().z() + dz);

    if (rangesIntersect(trajZRange, barrZRange)) {
      mylayers.push_back(firstPixelBarrelLayer());
    }
  }

  //negative fwd pixel

  for (auto infwd : firstPosPixelFwdLayer()) {
    pTsos = thePropagator.propagate(fastFts, infwd->surface());
    if (pTsos.isValid()) {
      Range nfwdRRange(infwd->specificSurface().innerRadius(), infwd->specificSurface().outerRadius());
      Range trajRRange(pTsos.globalPosition().perp() - dr, pTsos.globalPosition().perp() + dr);
      if (rangesIntersect(trajRRange, nfwdRRange)) {
        mylayers.push_back(infwd);
      }
    }
  }

  //positive fwd pixel
  for (auto ipfwd : firstPosPixelFwdLayer()) {
    pTsos = thePropagator.propagate(fastFts, ipfwd->surface());
    if (pTsos.isValid()) {
      Range pfwdRRange(ipfwd->specificSurface().innerRadius(), ipfwd->specificSurface().outerRadius());
      Range trajRRange(pTsos.globalPosition().perp() - dr, pTsos.globalPosition().perp() + dr);
      if (rangesIntersect(trajRRange, pfwdRRange)) {
        mylayers.push_back(ipfwd);
      }
    }
  }

  return mylayers;
}

const BarrelDetLayer* StartingLayerFinder::firstPixelBarrelLayer() const {
  checkPixelLayers();
  return theFirstPixelBarrelLayer;
}

const vector<const ForwardDetLayer*> StartingLayerFinder::firstNegPixelFwdLayer() const {
  checkPixelLayers();
  return theFirstNegPixelFwdLayer;
}

const vector<const ForwardDetLayer*> StartingLayerFinder::firstPosPixelFwdLayer() const {
  checkPixelLayers();
  return theFirstPosPixelFwdLayer;
}

void StartingLayerFinder::checkPixelLayers() const {
  if (!thePixelLayersValid) {
    const GeometricSearchTracker* theGeometricSearchTracker = theMeasurementTracker.geometricSearchTracker();

    theFirstPixelBarrelLayer = theGeometricSearchTracker->pixelBarrelLayers().front();
    theFirstNegPixelFwdLayer = theGeometricSearchTracker->negPixelForwardLayers();
    theFirstPosPixelFwdLayer = theGeometricSearchTracker->posPixelForwardLayers();
    thePixelLayersValid = true;
  }
}

#ifndef PIXELHITMATCHER_H
#define PIXELHITMATCHER_H

// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelHitMatcher
//
/**\class PixelHitMatcher EgammaElectronAlgos/PixelHitMatcher

 Description: Class to match an ECAL cluster to the pixel hits.
  Two compatible hits in the pixel layers are required.

 Implementation:
     future redesign
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "CLHEP/Vector/ThreeVector.h"
#include <vector>
#include <unordered_map>
#include <limits>

/** Class to match an ECAL cluster to the pixel hits.
 *  Two compatible hits in the pixel layers are required.
 */

class MeasurementTracker;
class TrackerGeometry;

namespace std {
  template <>
  struct hash<std::pair<const GeomDet*, GlobalPoint> > {
    std::size_t operator()(const std::pair<const GeomDet*, GlobalPoint>& g) const {
      auto h1 = std::hash<unsigned long long>()((unsigned long long)g.first);
      unsigned long long k;
      memcpy(&k, &g.second, sizeof(k));
      auto h2 = std::hash<unsigned long long>()(k);
      return h1 ^ (h2 << 1);
    }
  };
}  // namespace std

struct SeedWithInfo {
  const TrajectorySeed seed;
  const unsigned char hitsMask;
  const int subDet2;
  const float dRz2;
  const float dPhi2;
  const int subDet1;
  const float dRz1;
  const float dPhi1;
};

class PixelHitMatcher {
public:
  PixelHitMatcher(float phi1min,
                  float phi1max,
                  float phi2minB,
                  float phi2maxB,
                  float phi2minF,
                  float phi2maxF,
                  float z2minB,
                  float z2maxB,
                  float r2minF,
                  float r2maxF,
                  float rMinI,
                  float rMaxI,
                  bool useRecoVertex);

  void setES(const MagneticField*, const TrackerGeometry* trackerGeometry);

  std::vector<SeedWithInfo> operator()(const std::vector<const TrajectorySeedCollection*>& seedsV,
                                       const GlobalPoint& xmeas,
                                       const GlobalPoint& vprim,
                                       float energy,
                                       int charge) const;

  void set1stLayer(float dummyphi1min, float dummyphi1max);
  void set1stLayerZRange(float zmin1, float zmax1);
  void set2ndLayer(float dummyphi2minB, float dummyphi2maxB, float dummyphi2minF, float dummyphi2maxF);

private:
  BarrelMeasurementEstimator meas1stBLayer;
  BarrelMeasurementEstimator meas2ndBLayer;
  ForwardMeasurementEstimator meas1stFLayer;
  ForwardMeasurementEstimator meas2ndFLayer;
  std::unique_ptr<PropagatorWithMaterial> prop1stLayer;
  std::unique_ptr<PropagatorWithMaterial> prop2ndLayer;
  const MagneticField* theMagField;
  const TrackerGeometry* theTrackerGeometry;
  const bool useRecoVertex_;
};

#endif

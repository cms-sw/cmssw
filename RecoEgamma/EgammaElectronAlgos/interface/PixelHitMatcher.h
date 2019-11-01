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

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

/** Class to match an ECAL cluster to the pixel hits.
 *  Two compatible hits in the pixel layers are required.
 */

class TrackerGeometry;
class MagneticField;

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
  struct BarrelMeasurementEstimator {
    bool operator()(const GlobalPoint& vprim, const TrajectoryStateOnSurface& ts, const GlobalPoint& gp) const;

    float thePhiMin;
    float thePhiMax;
    float theZMin;
    float theZMax;
  };

  struct ForwardMeasurementEstimator {
    bool operator()(const GlobalPoint& vprim, const TrajectoryStateOnSurface& ts, const GlobalPoint& gp) const;

    float thePhiMin;
    float thePhiMax;
    float theRMin;
    float theRMax;
    const float theRMinI;
    const float theRMaxI;
  };

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

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
// $Id: PixelHitMatcher.h,v 1.17 2008/04/08 16:39:14 uberthon Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h" 

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchStartLayers.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h" 

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "CLHEP/Vector/ThreeVector.h"
#include <vector>

/** Class to match an ECAL cluster to the pixel hits.
 *  Two compatible hits in the pixel layers are required.
 */

class MeasurementTracker;
class MagneticField;
class GeometricSearchTracker;
class LayerMeasurements;
class TrackerGeometry;

class RecHitWithDist
{
 public: 

  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;

  RecHitWithDist(ConstRecHitPointer rh, float &dphi) : rh_(rh), dphi_(dphi)
    {}
  ConstRecHitPointer  recHit() const {return rh_;}
  float dPhi() const {return dphi_;}
  void invert() {dphi_*=-1.;}

 private:
  
   ConstRecHitPointer rh_;  
   float dphi_;

};

class PixelHitMatcher{  
 public:

  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  
  PixelHitMatcher(float phi1min, float phi1max, float phi2min, float phi2max, 
		  float z2minB, float z2maxB, float r2minF, float r2maxF,
		  float rMinI, float rMaxI, bool searchInTIDTEC);
		  
  virtual ~PixelHitMatcher();
  void setES(const MagneticField*, const MeasurementTracker *theMeasurementTracker, const TrackerGeometry *trackerGeometry);

  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > 
   compatibleHits(const GlobalPoint& xmeas, const GlobalPoint& vprim, float energy, float charge);
  std::vector<TrajectorySeed> 
   compatibleSeeds(edm::Handle<TrajectorySeedCollection> &seeds, const GlobalPoint& xmeas,
                   const GlobalPoint& vprim, float energy, float charge);
   
  std::vector<Hep3Vector> predicted1Hits();
  std::vector<Hep3Vector> predicted2Hits();

  void set1stLayer (float dummyphi1min, float dummyphi1max);
  void set1stLayerZRange (float zmin1, float zmax1);
  void set2ndLayer (float dummyphi2min, float dummyphi2max);
 
  float getVertex();

 private:

  RecHitContainer hitsInTrack;

  std::vector<Hep3Vector> pred1Meas;
  std::vector<Hep3Vector> pred2Meas; 
  FTSFromVertexToPointFactory myFTS;
  BarrelMeasurementEstimator meas1stBLayer;
  BarrelMeasurementEstimator meas2ndBLayer;
  ForwardMeasurementEstimator meas1stFLayer;
  ForwardMeasurementEstimator meas2ndFLayer;
  PixelMatchStartLayers startLayers;
  PropagatorWithMaterial *prop1stLayer;
  PropagatorWithMaterial *prop2ndLayer;
  const GeometricSearchTracker *theGeometricSearchTracker;
  const LayerMeasurements *theLayerMeasurements;
  const MagneticField* theMagField;
  const TrackerGeometry * theTrackerGeometry;

  float vertex_;

  bool searchInTIDTEC_;


};

#endif









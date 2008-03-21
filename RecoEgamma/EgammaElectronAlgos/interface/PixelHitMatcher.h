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
// $Id: PixelHitMatcher.h,v 1.14 2008/02/29 10:36:49 uberthon Exp $
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
		  float z2minB, float z2maxB,
		  float r2minF, float r2maxF,
		  float rMinI, float rMaxI, bool searchInTIDTEC) :
    //zmin1 and zmax1 are dummy at this moment, set from beamspot later
    meas1stBLayer(phi1min,phi1max,0.,0.), meas2ndBLayer(phi2min,phi2max,z2minB,z2maxB), 
    meas1stFLayer(phi1min,phi1max,0.,0.), meas2ndFLayer(phi2min,phi2max,r2minF,r2maxF),
    startLayers(),
    prop1stLayer(0), prop2ndLayer(0),theGeometricSearchTracker(0),theLayerMeasurements(0),vertex(0.),
    searchInTIDTEC_(searchInTIDTEC)
    {
      meas1stFLayer.setRRange(rMinI,rMaxI);
      meas2ndFLayer.setRRange(rMinI,rMaxI);
    }
  virtual ~PixelHitMatcher();
  void setES(const MagneticField*, const MeasurementTracker *theMeasurementTracker, const TrackerGeometry *trackerGeometry);

  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > compatibleHits(const GlobalPoint& xmeas,
									    const GlobalPoint& vprim,
									    float energy,
									    float charge);
  std::vector<TrajectorySeed> compatibleSeeds(edm::Handle<TrajectorySeedCollection> &trajectories, const GlobalPoint& xmeas,
									    const GlobalPoint& vprim,
									    float energy,
									    float charge);
  std::vector<Hep3Vector> predicted1Hits();
  std::vector<Hep3Vector> predicted2Hits();
  float getVertex();
 
  void set1stLayer (float dummyphi1min, float dummyphi1max)
    { 
      meas1stBLayer.setPhiRange(dummyphi1min,dummyphi1max) ;
      meas1stFLayer.setPhiRange(dummyphi1min,dummyphi1max) ;
    }

  void set1stLayerZRange (float zmin1, float zmax1)
    { 
      meas1stBLayer.setZRange(zmin1,zmax1);
      meas1stFLayer.setZRange(zmin1,zmax1);
    }

  void set2ndLayer (float dummyphi2min, float dummyphi2max)
    { 
                                         meas2ndBLayer.setPhiRange(dummyphi2min,dummyphi2max) ;
                                         meas2ndFLayer.setPhiRange(dummyphi2min,dummyphi2max) ;
				       }
 
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

  float vertex;

  bool searchInTIDTEC_;


};

#endif









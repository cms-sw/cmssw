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
// $Id: PixelHitMatcher.h,v 1.23 2009/05/27 07:31:22 fabiocos Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h" 

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchStartLayers.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h" 

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "CLHEP/Vector/ThreeVector.h"
#include <vector>
#include <limits>

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


class RecHitWithInfo
 {
  public: 

    typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
    typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
    typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  
    RecHitWithInfo( ConstRecHitPointer rh, int subDet =0,
       float dRz =std::numeric_limits<float>::infinity(),
       float dPhi =std::numeric_limits<float>::infinity() )
     : rh_(rh), subDet_(subDet), dRz_(dRz), dPhi_(dPhi) {}
      
    ConstRecHitPointer recHit() const { return rh_; }
    
    int subDet() { return subDet_ ; }
    float dRz() { return dRz_ ; }
    float dPhi() { return dPhi_ ; }

    void invert() { dPhi_*=-1. ; }
  
  private:
  
    ConstRecHitPointer rh_;  
    int subDet_ ;
    float dRz_ ;
    float dPhi_ ;
 } ;

class SeedWithInfo
 {
  public :
  
    SeedWithInfo( TrajectorySeed seed, int subDet2, float dRz2, float dPhi2 )
     : seed_(seed), subDet2_(subDet2), dRz2_(dRz2), dPhi2_(dPhi2) {}
     
    const TrajectorySeed & seed() { return seed_ ; }
    
    int subDet2() { return subDet2_ ; }
    float dRz2() { return dRz2_ ; }
    float dPhi2() { return dPhi2_ ; }
    
  private :
  
    TrajectorySeed seed_ ;
    int subDet2_ ;
    float dRz2_ ;
    float dPhi2_ ;
 } ;

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
  
  //   compatibleSeeds(edm::Handle<TrajectorySeedCollection> &seeds, const GlobalPoint& xmeas,
  std::vector<SeedWithInfo> 
  compatibleSeeds
    ( TrajectorySeedCollection * seeds, const GlobalPoint & xmeas,
      const GlobalPoint & vprim, float energy, float charge ) ;
   
  std::vector<CLHEP::Hep3Vector> predicted1Hits();
  std::vector<CLHEP::Hep3Vector> predicted2Hits();

  void set1stLayer (float dummyphi1min, float dummyphi1max);
  void set1stLayerZRange (float zmin1, float zmax1);
  void set2ndLayer (float dummyphi2min, float dummyphi2max);
 
  float getVertex();

 private:

  RecHitContainer hitsInTrack;

  std::vector<CLHEP::Hep3Vector> pred1Meas;
  std::vector<CLHEP::Hep3Vector> pred2Meas; 
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
  std::vector<std::pair<const GeomDet*, TrajectoryStateOnSurface> >  mapTsos_;
  std::vector<std::pair<std::pair<const GeomDet*,GlobalPoint>,  TrajectoryStateOnSurface> >  mapTsos2_;
};
#endif









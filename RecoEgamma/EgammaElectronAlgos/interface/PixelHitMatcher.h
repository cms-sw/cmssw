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
// $Id: PixelHitMatcher.h,v 1.3 2007/02/05 17:53:51 uberthon Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h" 
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchStartLayers.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
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
class NavigationSchool;

class RecHitWithDist
{
 public: 

  //RC
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;

  //UB change place??
  //RC RecHitWithDist(const TSiPixelRecHit &rh, float &dphi) : rh_(rh), dphi_(dphi)
  RecHitWithDist(ConstRecHitPointer rh, float &dphi) : rh_(rh), dphi_(dphi)
    {}
  //RC const TSiPixelRecHit & recHit() const {return rh_;}
  ConstRecHitPointer  recHit() const {return rh_;}
  float dPhi() const {return dphi_;}
  void invert() {dphi_*=-1.;}

 private:
  
   ConstRecHitPointer rh_;  
   float dphi_;

};

class PixelHitMatcher{  
 public:
  //RC
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  

  PixelHitMatcher(float phi1min, float phi1max, float phi2min, float phi2max, 
		  float z1min, float z1max, float z2min, float z2max) :
    phi1min(phi1min), phi1max(phi1max), phi2min(phi2min), phi2max(phi2max), 
    z1min(z1min), z1max(z1max), z2min(z2min), z2max(z2max), 
    meas1stBLayer(phi1min,phi1max,z1min,z1max), meas2ndBLayer(phi2min,phi2max,z2min,z2max), 
    meas1stFLayer(phi1min,phi1max,z1min,z1max), meas2ndFLayer(phi2min,phi2max,z2min,z2max),
    startLayers(),
    //    prop1stLayer(oppositeToMomentum,.511), prop2ndLayer(alongMomentum,.511),//depends on event?
    prop1stLayer(0), prop2ndLayer(0),//depends on event?
    //      theNavigationSchool( new SimpleNavigationSchool), vertex(0.) {} //depends on event??
    theNavigationSchool(0),theGeometricSearchTracker(0),theLayerMeasurements(0),
    vertex(0.) {}
  virtual ~PixelHitMatcher();
  void setES(const MagneticField*, const MeasurementTracker *theMeasurementTracker);

  //RC vector<pair<RecHitWithDist,TSiPixelRecHit> > compatibleHits(const GlobalPoint& xmeas,
  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > compatibleHits(const GlobalPoint& xmeas,
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
  void set2ndLayer (float dummyphi2min, float dummyphi2max)
                                       { 
                                         meas2ndBLayer.setPhiRange(dummyphi2min,dummyphi2max) ;
                                         meas2ndFLayer.setPhiRange(dummyphi2min,dummyphi2max) ;
				       }
 
 private:
  //vector<TSiPixelRecHit> hitsInTrack;
  RecHitContainer hitsInTrack;

  float phi1min, phi1max, phi2min, phi2max, z1min, z1max, z2min, z2max;
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
  NavigationSchool* theNavigationSchool;
  const GeometricSearchTracker *theGeometricSearchTracker;
  const LayerMeasurements *theLayerMeasurements;
  const MagneticField* theMagField;
  float vertex;


};

#endif









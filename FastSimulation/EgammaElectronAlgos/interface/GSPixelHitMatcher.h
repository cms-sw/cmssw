#ifndef GSPIXELHITMATCHER_H
#define GSPIXELHITMATCHER_H

// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      GSPixelHitMatcher
// 
/**\class GSPixelHitMatcher EgammaElectronAlgos/GSPixelHitMatcher

 Description: Class to match an ECAL cluster to the pixel hits.
  Two compatible hits in the pixel layers are required.

 Implementation:
     future redesign
*/
//
// Original Author:  Patrick Janot.
//
//

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include <vector>

/** Class to match an ECAL cluster to the pixel hits.
 *  Two compatible hits in the pixel layers are required.
 */

class TrackerGeometry;
class GeometricSearchTracker;
class TrackerInteractionGeometry;
class TrackerLayer;
class TrackingRecHit;
class ParticlePropagator;
class MagneticFieldMap;

class GSPixelHitMatcher{  

 public:
  //RC
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
  

  GSPixelHitMatcher(float,float,float,float,float,float,float,float,float,float);

  virtual ~GSPixelHitMatcher();

  void setES(const MagneticFieldMap* aFieldMap, 
	     const TrackerGeometry* aTrackerGeometry, 
	     const GeometricSearchTracker* geomSearchTracker,
	     const TrackerInteractionGeometry* interactionGeometry);

  std::vector< std::pair<ConstRecHitPointer,ConstRecHitPointer> > 
    compatibleHits(const GlobalPoint& xmeas,
		   const GlobalPoint& vprim,
		   float energy,
		   std::vector<ConstRecHitPointer>& thePixelRecHits);

  float getVertex();

  void set1stLayer (float ephimin, float ephimax,
		    float pphimin, float pphimax) {
  
    ephi1min = ephimin;
    ephi1max = ephimax;
    pphi1min = pphimin;
    pphi1max = pphimax;

  }

  void set2ndLayer (float phimin, float phimax) { 
    phi2min = phimin;
    phi2max = phimax;
  }

  bool isASeed(const ParticlePropagator& myElec,
	       const ParticlePropagator& myPosi,
	       double rCluster,
	       double zCluster,
	       ConstRecHitPointer hit1,
	       ConstRecHitPointer hit2);
  
  bool propagateToLayer(ParticlePropagator& myPart,
			GlobalPoint& theHit,
			double zVertex,
			double phimin, 
			double phimax,
			unsigned layer);
  
  double zVertex(double zCluster, 
		 double rCluster,
		 GlobalPoint& theHit);
 
  bool zCompatible(double zVertex, double zPrior, 
		   double zmin, double zmax,
		   bool barrel); 

 private:

  RecHitContainer hitsInTrack;

  float ephi1min, ephi1max;
  float pphi1min, pphi1max;
  float phi2min, phi2max;
  float z1min, z1max, z2min, z2max;
  const TrackerGeometry* theTrackerGeometry;
  const MagneticField* theMagneticField;
  const GeometricSearchTracker* theGeomSearchTracker;
  const TrackerInteractionGeometry* _theGeometry; 
  const MagneticFieldMap* theFieldMap;
  std::vector<const TrackerLayer*> thePixelLayers;
  float vertex;


};

#endif









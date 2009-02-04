#ifndef FastPixelHitMatcher_H
#define FastPixelHitMatcher_H

// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      FastPixelHitMatcher
//
/**\class FastPixelHitMatcher EgammaElectronAlgos/FastPixelHitMatcher

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
class TrackerRecHit;
class TrackingRecHit;
class ParticlePropagator;
class MagneticFieldMap;

class FastPixelHitMatcher{

 public:
  //RC
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;


  FastPixelHitMatcher(float,float,float,float,float,float,float,float,float,float,float,float,bool);

  virtual ~FastPixelHitMatcher();

  void setES(const MagneticFieldMap* aFieldMap,
	     const TrackerGeometry* aTrackerGeometry,
	     const GeometricSearchTracker* geomSearchTracker,
	     const TrackerInteractionGeometry* interactionGeometry);

  std::vector< std::pair<ConstRecHitPointer,ConstRecHitPointer> >
    compatibleHits(const GlobalPoint& xmeas,
		   const GlobalPoint& vprim,
		   float energy,
		   std::vector<TrackerRecHit>& theHits);

  inline double getVertex() { return vertex; }

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
	       const GlobalPoint& theVertex,
	       double rCluster,
	       double zCluster,
	       const TrackerRecHit& hit1,
	       const TrackerRecHit& hit2);

  bool propagateToLayer(ParticlePropagator& myPart,
			const GlobalPoint& theVertex,
			GlobalPoint& theHit,
			double phimin,
			double phimax,
			unsigned layer);

  double zVertex(double zCluster,
		 double rCluster,
		 GlobalPoint& theHit);

  bool checkRZCompatibility(double zCluster,double rCluster,
			    double zVertex,
			    float rzMin, float rzMax,
			    GlobalPoint& theHit,
			    bool forward);


  void set1stLayerZRange(double zmin1, double zmax1) {
    z1max = zmax1;
    z1min = zmin1;
  }

 private:

  RecHitContainer hitsInTrack;

  float ephi1min, ephi1max;
  float pphi1min, pphi1max;
  float phi2min, phi2max;
  float z1min, z1max;
  float z2minB, z2maxB;
  float r2minF, r2maxF;
  float rMinI, rMaxI;
  bool searchInTIDTEC;

  const TrackerGeometry* theTrackerGeometry;
  const MagneticField* theMagneticField;
  const GeometricSearchTracker* theGeomSearchTracker;
  const TrackerInteractionGeometry* _theGeometry;
  const MagneticFieldMap* theFieldMap;
  std::vector<const TrackerLayer*> thePixelLayers;
  double vertex;


};

#endif









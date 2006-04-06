#ifndef _TrackerLayer_H_
#define _TrackerLayer_H_

#include "Geometry/Surface/interface/BoundSurface.h"

/** A class that gives some properties of the Tracker Layers in FAMOS
 */

class TrackerLayer {
public:
  
  /// constructor from private members
  TrackerLayer(BoundSurface* theSurface,
	       bool isForward,
	       unsigned theLayerNumber    = 0,
	       double theResolutionAlongX = 0.,
	       double theResolutionAlongY = 0.,
	       double theHitEfficiency    = 1. ) :
    theSurface(theSurface), 
    isForward(isForward),
    theLayerNumber(theLayerNumber),
    theResolutionAlongX(theResolutionAlongX),
    theResolutionAlongY(theResolutionAlongY),
    theHitEfficiency(theHitEfficiency)
   { 
     isSensitive = (theLayerNumber!=0);
     theFirstRing = 0;
     theLastRing = 0;
   }

  TrackerLayer(BoundSurface* theSurface,
	       int theLayerNumber,
	       int theFirstRing, 
	       int theLastRing ) :
    theSurface(theSurface), 
    theLayerNumber(theLayerNumber),
    theFirstRing(theFirstRing),
    theLastRing(theLastRing) 
   { 
     isSensitive = true;
     isForward = true;
     theResolutionAlongX = 0.;
     theResolutionAlongY = 0.;
     theHitEfficiency = 1.;
   }

  /// Is the layer sensitive ?
  inline bool sensitive() const { return isSensitive; }

  /// Is the layer forward ?
  inline bool forward() const { return isForward; }

  /// Returns the surface
  inline const BoundSurface& surface() const { return *theSurface; }

  /// Returns the layer number  
  inline unsigned layerNumber() const { return theLayerNumber; }

  /// Returns the first ring  
  inline unsigned firstRing() const { return theFirstRing; }

  /// Returns the lasst ring  
  inline unsigned lastRing() const { return theLastRing; }

  /// Returns the resolution along x in cm (local coordinates)
  inline double resolutionAlongxInCm() const { return theResolutionAlongX; }

  /// Returns the resolution along y in cm(local coordinates)
  inline double resolutionAlongyInCm() const { return theResolutionAlongY; }

  /// Returns the hit reconstruction efficiency
  inline double hitEfficiency() const { return theHitEfficiency; }

private:

  BoundSurface* theSurface;
  bool isForward;
  unsigned theLayerNumber;
  unsigned theFirstRing;
  unsigned theLastRing;
  double theResolutionAlongX;
  double theResolutionAlongY;
  double theHitEfficiency;
  bool isSensitive;

};
#endif


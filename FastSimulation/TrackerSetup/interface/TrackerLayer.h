#ifndef _TrackerLayer_H_
#define _TrackerLayer_H_

#include "Geometry/Surface/interface/BoundSurface.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"

/** A class that gives some properties of the Tracker Layers in FAMOS
 */

#include <iostream>

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
     isSensitive = (theLayerNumber<100);
     theFirstRing = 0;
     theLastRing = 0;
     if ( isForward ) { 
       theDisk = dynamic_cast<BoundDisk*>(theSurface);
       theCylinder = 0;
     } else {
       theCylinder = dynamic_cast<BoundCylinder*>(theSurface);
       theDisk = 0;
     }

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
     theDisk = dynamic_cast<BoundDisk*>(theSurface);
     theCylinder = 0;
   }



  /// Copy constructor
  TrackerLayer(const TrackerLayer& other) :

    theSurface (other.theSurface),
    theDisk (other.theDisk),
    theCylinder (other.theCylinder),
    isForward (other.isForward),
    theLayerNumber (other.theLayerNumber),
    theFirstRing (other.theFirstRing),
    theLastRing (other.theLastRing),
    theResolutionAlongX (other.theResolutionAlongX),
    theResolutionAlongY (other.theResolutionAlongY),
    theHitEfficiency (other.theHitEfficiency),
    isSensitive (other.isSensitive) {
    
  }
    
  /// Is the layer sensitive ?
  inline bool sensitive() const { return isSensitive; }

  /// Is the layer forward ?
  inline bool forward() const { return isForward; }

  /// Returns the surface
  inline const BoundSurface& surface() const { return *theSurface; }

  /// Returns the cylinder
  inline BoundCylinder* cylinder() const { return theCylinder; }

  /// Returns the surface
  inline BoundDisk* disk() const { return theDisk; }

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
  BoundDisk* theDisk;
  BoundCylinder* theCylinder;
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


#ifndef _TrackerLayer_H_
#define _TrackerLayer_H_

#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

/** A class that gives some properties of the Tracker Layers in FAMOS
 */

class TrackerLayer {
public:
  
  /// constructor from private members
  TrackerLayer(BoundSurface* theSurface,
	       bool isForward,
	       unsigned int theLayerNumber,
	       double theModuleThickness = 0.,
	       double theResolutionAlongX = 0.,
	       double theResolutionAlongY = 0.,
	       double theHitEfficiency    = 1. ) :
    theSurface(theSurface), 
    isForward(isForward),
    theLayerNumber(theLayerNumber),
    theResolutionAlongX(theResolutionAlongX),
    theResolutionAlongY(theResolutionAlongY),
    theHitEfficiency(theHitEfficiency),
    theModuleThickness(theModuleThickness)
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
	       unsigned int theLayerNumber,
	       double theModuleThickness, 
	       unsigned int theFirstRing, 
	       unsigned int theLastRing) :
    theSurface(theSurface), 
    theLayerNumber(theLayerNumber),
    theFirstRing(theFirstRing),
    theLastRing(theLastRing),
    theModuleThickness(theModuleThickness)
   { 
     isSensitive = true;
     isForward = true;
     theResolutionAlongX = 0.;
     theResolutionAlongY = 0.;
     theHitEfficiency = 1.;
     theDisk = dynamic_cast<BoundDisk*>(theSurface);
     theCylinder = 0;
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
  inline unsigned int layerNumber() const { return theLayerNumber; }

  /// Returns the first ring  
  inline unsigned int firstRing() const { return theFirstRing; }

  /// Returns the lasst ring  
  inline unsigned int lastRing() const { return theLastRing; }

  /// Returns the resolution along x in cm (local coordinates)
  inline double resolutionAlongxInCm() const { return theResolutionAlongX; }

  /// Returns the resolution along y in cm(local coordinates)
  inline double resolutionAlongyInCm() const { return theResolutionAlongY; }

  /// Returns the hit reconstruction efficiency
  inline double hitEfficiency() const { return theHitEfficiency; }

  /// Returns the sensitive module thickness
  inline double moduleThickness() const { return theModuleThickness; }

private:

  BoundSurface* theSurface;
  BoundDisk* theDisk;
  BoundCylinder* theCylinder;
  bool isForward;
  unsigned int theLayerNumber;
  unsigned int theFirstRing;
  unsigned int theLastRing;
  double theResolutionAlongX;
  double theResolutionAlongY;
  double theHitEfficiency;
  double theModuleThickness;
  bool isSensitive;

};
#endif


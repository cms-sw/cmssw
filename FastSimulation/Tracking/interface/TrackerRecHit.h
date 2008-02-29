#ifndef FastSimulation_Tracking_TrackerRecHit_H_
#define FastSimulation_Tracking_TrackerRecHit_H_

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
//#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
//#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
//#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include <vector>

/** A class that gives some properties of the Tracker Layers in FAMOS
 */

class TrackerRecHit {
public:
  

  /// Default Constructor
  TrackerRecHit() :
    theHit(0),
    theGeomDet(0),
    theSubDetId(0),
    theLayerNumber(0),
    theRingNumber(0), 
    theLocalError(0.),
    theLargerError(0.),
    forward(false) {}
  
  /// Constructor from private members
    //  TrackerRecHit(const SiTrackerGSRecHit2D* theHit, 
    //		const TrackerGeometry* theGeometry);
  TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
		const TrackerGeometry* theGeometry);

  /// The Hit itself
    //  const SiTrackerGSRecHit2D* hit() const { return theHit; }
  const SiTrackerGSMatchedRecHit2D* hit() const { return theHit; }

  /// The subdet Id
  inline unsigned int subDetId() const { return theSubDetId; }

  /// The Layer Number
  inline unsigned int layerNumber() const { return theLayerNumber; }

  /// The Ring Number
  inline unsigned int ringNumber() const { return theRingNumber; }

  /// Is it a forward hit ?
  inline bool isForward() const { return forward; }

  /// The GeomDet
  inline const GeomDet* geomDet() const { return theGeomDet; }

  /// The global position
  inline GlobalPoint globalPosition() const { 
    return theGeomDet->surface().toGlobal(theHit->localPosition());
  }

  /// The local position
  inline LocalPoint localPosition() const { return theHit->localPosition(); }

  /// Check if the hit is on one of the requested detector
  bool isOnRequestedDet(const std::vector<unsigned int>& whichDet) const;

  /// Check if two hits are on the same layer of the same subdetector
  inline bool isOnTheSameLayer(const TrackerRecHit& other) const {
    
    return 
      theSubDetId == other.subDetId() && 
      theLayerNumber == other.layerNumber();
  }

  // The smaller local error
  double localError() { 

    // Check if it has been already computed
    if ( theLocalError != 0. ) return theLocalError;

    // Otherwise, compute it!
    double xx = theHit->localPositionError().xx();
    double yy = theHit->localPositionError().yy();
    double xy = theHit->localPositionError().xy();
    double delta = std::sqrt((xx-yy)*(xx-yy)+4.*xy*xy);
    theLocalError = 0.5 * (xx+yy-delta);
    return theLocalError;

  }
  
  // The larger local error
  double largerError() { 

    // Check if it has been already computed
    if ( theLargerError != 0. ) return theLargerError;

    // Otherwise, compute it!
    double xx = theHit->localPositionError().xx();
    double yy = theHit->localPositionError().yy();
    double xy = theHit->localPositionError().xy();
    double delta = std::sqrt((xx-yy)*(xx-yy)+4.*xy*xy);
    theLargerError = 0.5 * (xx+yy+delta);
    return theLargerError;

  }
  
  inline bool operator!=(const TrackerRecHit& aHit) const {
    return 
      aHit.geomDet() != this->geomDet() ||
      aHit.hit()->localPosition().x() != this->hit()->localPosition().x() ||
      aHit.hit()->localPosition().y() != this->hit()->localPosition().y() ||
      aHit.hit()->localPosition().z() != this->hit()->localPosition().z();
  }

 private:
  
  // const SiTrackerGSRecHit2D* theHit;
  const SiTrackerGSMatchedRecHit2D* theHit;
   const GeomDet* theGeomDet;
  unsigned int theSubDetId; 
  unsigned int theLayerNumber;
  unsigned int theRingNumber;
  double theLocalError;
  double theLargerError;
  bool forward;

};
#endif


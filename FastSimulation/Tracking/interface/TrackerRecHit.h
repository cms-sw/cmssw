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
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h" 
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
    theLocalError(0.) {}
  /// constructor from private members
  TrackerRecHit(const SiTrackerGSRecHit2D* theHit, const TrackerGeometry* theGeometry) :
    theHit(theHit),
    theSubDetId(0),
    theLayerNumber(0),
    theRingNumber(0),
    theLocalError(0.)
    { 
      const DetId& theDetId = theHit->geographicalId();
      theGeomDet = theGeometry->idToDet(theDetId);
      theSubDetId = theDetId.subdetId(); 
      if ( theSubDetId == StripSubdetector::TIB) { 
       TIBDetId tibid(theDetId.rawId()); 
       theLayerNumber = tibid.layer();
     } else if ( theSubDetId ==  StripSubdetector::TOB ) { 
       TOBDetId tobid(theDetId.rawId()); 
       theLayerNumber = tobid.layer();
     } else if ( theSubDetId ==  StripSubdetector::TID) { 
       TIDDetId tidid(theDetId.rawId());
       theLayerNumber = tidid.wheel();
       theRingNumber = tidid.ring();
     } else if ( theSubDetId ==  StripSubdetector::TEC ) { 
       TECDetId tecid(theDetId.rawId()); 
       theLayerNumber = tecid.wheel(); 
       theRingNumber = tecid.ring();
     } else if ( theSubDetId ==  PixelSubdetector::PixelBarrel ) { 
       PXBDetId pxbid(theDetId.rawId()); 
       theLayerNumber = pxbid.layer();  
     } else if ( theSubDetId ==  PixelSubdetector::PixelEndcap ) { 
       PXFDetId pxfid(theDetId.rawId()); 
       theLayerNumber = pxfid.disk();  
     }


   }

  /// The Hit itself
  const SiTrackerGSRecHit2D* hit() const { return theHit; }

  /// The subdet Id
  inline unsigned int subDetId() const { return theSubDetId; }

  /// The Layer Number
  inline unsigned int layerNumber() const { return theLayerNumber; }

  /// The Ring Number
  inline unsigned int ringNumber() const { return theRingNumber; }

  /// The GeomDet
  inline const GeomDet* geomDet() const { return theGeomDet; }

  /// The global position
  inline GlobalPoint globalPosition() const { 
    return theGeomDet->surface().toGlobal(theHit->localPosition());
  }

  /// The local position
  inline LocalPoint localPosition() const { return theHit->localPosition(); }

  /// Check if the hit is on one of the requested detector
  bool isOnRequestedDet(const std::vector<unsigned int>& whichDet) { 

    bool isOnDet = false;

    for ( unsigned idet=0; idet<whichDet.size(); ++idet ) {
      
      switch ( whichDet[idet] ) { 
	
      case 1: 
	//Pixel Barrel
	isOnDet =  theSubDetId==1;
	break;
	
      case 2: 
	//Pixel Disks
	isOnDet = theSubDetId==2;
	break;
	
      case 3:
	//Inner Barrel
	isOnDet = theSubDetId==3 && theLayerNumber < 3;
	break;
	
      case 4:
	//Inner Disks
	isOnDet = theSubDetId==4 && theRingNumber < 3;
	break;
	
      case 5:
	//Outer Barrel
	isOnDet = false;
	break;
	
      case 6:
	//Tracker EndCap
	isOnDet = theSubDetId==6 && theLayerNumber < 4 && theRingNumber < 3;
	break;
	
      default:
	// Should not happen
	isOnDet = false;
	break;
	
      }

      if ( isOnDet ) break;

    }

    return isOnDet;
  }

  // The smallest local error
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
  
  inline bool operator!=(const TrackerRecHit& aHit) const {
    std::cout << "The geom Dets = " << aHit.geomDet() 
	      << " " << this->geomDet() 
	      << std::endl
	      << "The positions = " << aHit.hit()->localPosition() 
	      << " " << this->hit()->localPosition() 
	      << std::endl;

    return 
      aHit.geomDet() != this->geomDet() ||
      aHit.hit()->localPosition().x() != this->hit()->localPosition().x() ||
      aHit.hit()->localPosition().y() != this->hit()->localPosition().y() ||
      aHit.hit()->localPosition().z() != this->hit()->localPosition().z();
  }

 private:
  
  const SiTrackerGSRecHit2D* theHit;
  const GeomDet* theGeomDet;
  unsigned int theSubDetId; 
  unsigned int theLayerNumber;
  unsigned int theRingNumber;
  double theLocalError;

};
#endif


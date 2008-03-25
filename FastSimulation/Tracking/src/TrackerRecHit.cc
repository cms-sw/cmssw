#include "FastSimulation/Tracking/interface/TrackerRecHit.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

TrackerRecHit::TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
			     const TrackerGeometry* theGeometry) :
  theSplitHit(0),
  theMatchedHit(theHit),
  theSubDetId(0),
  theLayerNumber(0),
  theRingNumber(0),
  theCylinderNumber(0),
  theLocalError(0.),
  theLargerError(0.)
     
{ 
  init(theGeometry);
}

TrackerRecHit::TrackerRecHit(const SiTrackerGSRecHit2D* theHit, 
			     const TrackerGeometry* theGeometry) :
  theSplitHit(theHit),
  theMatchedHit(0),
  theSubDetId(0),
  theLayerNumber(0),
  theRingNumber(0),
  theCylinderNumber(0),
  theLocalError(0.),
  theLargerError(0.)
     
{ 
  init(theGeometry);
}

void
TrackerRecHit::init(const TrackerGeometry* theGeometry) { 

  const DetId& theDetId = hit()->geographicalId();
  theGeomDet = theGeometry->idToDet(theDetId);
  theSubDetId = theDetId.subdetId(); 
  if ( theSubDetId == StripSubdetector::TIB) { 
    TIBDetId tibid(theDetId.rawId()); 
    theLayerNumber = tibid.layer();
    theCylinderNumber = TrackerInteractionGeometry::TIB+theLayerNumber;
    forward = false;
  } else if ( theSubDetId ==  StripSubdetector::TOB ) { 
    TOBDetId tobid(theDetId.rawId()); 
    theLayerNumber = tobid.layer();
    theCylinderNumber = TrackerInteractionGeometry::TOB+theLayerNumber;
    forward = false;
  } else if ( theSubDetId ==  StripSubdetector::TID) { 
    TIDDetId tidid(theDetId.rawId());
    theLayerNumber = tidid.wheel();
    theCylinderNumber = TrackerInteractionGeometry::TID+theLayerNumber;
    theRingNumber = tidid.ring();
    forward = true;
  } else if ( theSubDetId ==  StripSubdetector::TEC ) { 
    TECDetId tecid(theDetId.rawId()); 
    theLayerNumber = tecid.wheel(); 
    theCylinderNumber = TrackerInteractionGeometry::TEC+theLayerNumber;
    theRingNumber = tecid.ring();
    forward = true;
  } else if ( theSubDetId ==  PixelSubdetector::PixelBarrel ) { 
    PXBDetId pxbid(theDetId.rawId()); 
    theLayerNumber = pxbid.layer(); 
    theCylinderNumber = TrackerInteractionGeometry::PXB+theLayerNumber;
    forward = false;
  } else if ( theSubDetId ==  PixelSubdetector::PixelEndcap ) { 
    PXFDetId pxfid(theDetId.rawId()); 
    theLayerNumber = pxfid.disk();  
    theCylinderNumber = TrackerInteractionGeometry::PXD+theLayerNumber;
    forward = true;
  }
  
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<unsigned int>& whichDet) const { 
  
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
      isOnDet = theSubDetId==3 && theLayerNumber < 4;
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



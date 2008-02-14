#include "FastSimulation/Tracking/interface/TrackerRecHit.h"


//TrackerRecHit::TrackerRecHit(const SiTrackerGSRecHit2D* theHit, 
//		     const TrackerGeometry* theGeometry) :
TrackerRecHit::TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
			     const TrackerGeometry* theGeometry) :
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


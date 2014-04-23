#include "FastSimulation/Tracking/interface/TrackerRecHit.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

TrackerRecHit::TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
			     const TrackerGeometry* theGeometry,
			     const TrackerTopology* tTopo) :
  theSplitHit(0),
  theMatchedHit(theHit),
  theSubDetId(0),
  theLayerNumber(0),
  theRingNumber(0),
  theCylinderNumber(0),
  theLocalError(0.),
  theLargerError(0.)
     
{ 
  init(theGeometry, tTopo);
}

TrackerRecHit::TrackerRecHit(const SiTrackerGSRecHit2D* theHit, 
			     const TrackerGeometry* theGeometry,
			     const TrackerTopology* tTopo ) :
  theSplitHit(theHit),
  theMatchedHit(0),
  theSubDetId(0),
  theLayerNumber(0),
  theRingNumber(0),
  theCylinderNumber(0),
  theLocalError(0.),
  theLargerError(0.)
     
{ 
  init(theGeometry,tTopo);
}

void
TrackerRecHit::init(const TrackerGeometry* theGeometry, const TrackerTopology *tTopo) { 

  const DetId& theDetId = hit()->geographicalId();
  theGeomDet = theGeometry->idToDet(theDetId);
  theSubDetId = theDetId.subdetId(); 
  if ( theSubDetId == StripSubdetector::TIB) { 
     
    theLayerNumber = tTopo->tibLayer(theDetId);
    theCylinderNumber = TrackerInteractionGeometry::TIB+theLayerNumber;
    forward = false;
  } else if ( theSubDetId ==  StripSubdetector::TOB ) { 
     
    theLayerNumber = tTopo->tobLayer(theDetId);
    theCylinderNumber = TrackerInteractionGeometry::TOB+theLayerNumber;
    forward = false;
  } else if ( theSubDetId ==  StripSubdetector::TID) { 
    
    theLayerNumber = tTopo->tidWheel(theDetId);
    theCylinderNumber = TrackerInteractionGeometry::TID+theLayerNumber;
    theRingNumber = tTopo->tidRing(theDetId);
    forward = true;
  } else if ( theSubDetId ==  StripSubdetector::TEC ) { 
     
    theLayerNumber = tTopo->tecWheel(theDetId); 
    theCylinderNumber = TrackerInteractionGeometry::TEC+theLayerNumber;
    theRingNumber = tTopo->tecRing(theDetId);
    forward = true;
  } else if ( theSubDetId ==  PixelSubdetector::PixelBarrel ) { 
     
    theLayerNumber = tTopo->pxbLayer(theDetId); 
    theCylinderNumber = TrackerInteractionGeometry::PXB+theLayerNumber;
    forward = false;
  } else if ( theSubDetId ==  PixelSubdetector::PixelEndcap ) { 
     
    theLayerNumber = tTopo->pxfDisk(theDetId);  
    theCylinderNumber = TrackerInteractionGeometry::PXD+theLayerNumber;
    forward = true;
  }
  
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<TrajectorySeedProducer::LayerSpec> >& theLayersInSets) const{ 
  
  for(unsigned int i=0; i<theLayersInSets.size(); ++i) {
    if(theLayersInSets[i][0].subDet==theSubDetId && theLayersInSets[i][0].idLayer==theLayerNumber) return true;
  }

  return false;
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<TrajectorySeedProducer::LayerSpec> >& theLayersInSets,  const TrackerRecHit& theSeedHitSecond) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0].subDet==theSubDetId && theLayersInSets[i][0].idLayer==theLayerNumber &&
        theLayersInSets[i][1].subDet==theSeedHitSecond.subDetId() && theLayersInSets[i][1].idLayer==theSeedHitSecond.layerNumber()
      ) return true;
  }
  return false;
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<TrajectorySeedProducer::LayerSpec> >& theLayersInSets,  const TrackerRecHit& theSeedHitSecond, const TrackerRecHit& theSeedHitThird) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0].subDet==theSubDetId && theLayersInSets[i][0].idLayer==theLayerNumber &&
        theLayersInSets[i][1].subDet==theSeedHitSecond.subDetId() && theLayersInSets[i][1].idLayer==theSeedHitSecond.layerNumber() &&
        theLayersInSets[i][2].subDet==theSeedHitThird.subDetId() && theLayersInSets[i][2].idLayer==theSeedHitThird.layerNumber() 
      ) return true;
  }
  return false;
}


bool
//TrackerRecHit::isOnRequestedDet(const std::vector<unsigned int>& whichDet) const { 
TrackerRecHit::isOnRequestedDet(const std::vector<unsigned int>& whichDet, const std::string& seedingAlgo) const { 
  
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
      if(seedingAlgo == "TobTecLayerPairs"){
	isOnDet = theSubDetId==5 && theLayerNumber <3;
      }else {
	isOnDet = false;
      }
      break;
      
    case 6:
      //Tracker EndCap
      if(seedingAlgo == "PixelLessPairs"){
	isOnDet = theSubDetId==6 && theLayerNumber < 6 && theRingNumber < 3;
      }else if (seedingAlgo == "TobTecLayerPairs"){
	//	isOnDet = theSubDetId==6 && theLayerNumber < 8 && theRingNumber < 5;
	isOnDet = theSubDetId==6 && theLayerNumber < 8 && theRingNumber == 5;
      } else if (seedingAlgo == "MixedTriplets"){ 
	//	isOnDet = theSubDetId==6 && theLayerNumber == 2 && theRingNumber == 1;
	isOnDet = theSubDetId==6 && theLayerNumber < 4 && theRingNumber == 1;
      } else {
	isOnDet = theSubDetId==6;
	std::cout << "DEBUG - this should never happen" << std::endl;
      }

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






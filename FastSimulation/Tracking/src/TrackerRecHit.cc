#include "FastSimulation/Tracking/interface/TrackerRecHit.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

TrackerRecHit::TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
			     const TrackerGeometry* theGeometry,
			     const TrackerTopology* tTopo) :
  theSplitHit(0),
  theMatchedHit(theHit),
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
  seedingLayer.subDet = theDetId.subdetId(); 
  if ( seedingLayer.subDet == StripSubdetector::TIB) { 
     
    seedingLayer.idLayer = tTopo->tibLayer(theDetId);
    theCylinderNumber = TrackerInteractionGeometry::TIB+seedingLayer.idLayer;
    forward = false;
  } else if ( seedingLayer.subDet ==  StripSubdetector::TOB ) { 
     
    seedingLayer.idLayer = tTopo->tobLayer(theDetId);
    theCylinderNumber = TrackerInteractionGeometry::TOB+seedingLayer.idLayer;
    forward = false;
  } else if ( seedingLayer.subDet ==  StripSubdetector::TID) { 
    
    seedingLayer.idLayer = tTopo->tidWheel(theDetId);
    theCylinderNumber = TrackerInteractionGeometry::TID+seedingLayer.idLayer;
    theRingNumber = tTopo->tidRing(theDetId);
    forward = true;
  } else if ( seedingLayer.subDet ==  StripSubdetector::TEC ) { 
     
    seedingLayer.idLayer = tTopo->tecWheel(theDetId); 
    theCylinderNumber = TrackerInteractionGeometry::TEC+seedingLayer.idLayer;
    theRingNumber = tTopo->tecRing(theDetId);
    forward = true;
  } else if ( seedingLayer.subDet ==  PixelSubdetector::PixelBarrel ) { 
     
    seedingLayer.idLayer = tTopo->pxbLayer(theDetId); 
    theCylinderNumber = TrackerInteractionGeometry::PXB+seedingLayer.idLayer;
    forward = false;
  } else if ( seedingLayer.subDet ==  PixelSubdetector::PixelEndcap ) { 
     
    seedingLayer.idLayer = tTopo->pxfDisk(theDetId);  
    theCylinderNumber = TrackerInteractionGeometry::PXD+seedingLayer.idLayer;
    forward = true;
  }
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<LayerSpec> >& theLayersInSets) const{ 
  
  for(unsigned int i=0; i<theLayersInSets.size(); ++i) {
    if(theLayersInSets[i][0].subDet==seedingLayer.subDet && theLayersInSets[i][0].idLayer==seedingLayer.idLayer) return true;
  }

  return false;
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<LayerSpec> >& theLayersInSets,  const TrackerRecHit& theSeedHitSecond) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0].subDet==seedingLayer.subDet && theLayersInSets[i][0].idLayer==seedingLayer.idLayer &&
        theLayersInSets[i][1].subDet==theSeedHitSecond.subDetId() && theLayersInSets[i][1].idLayer==theSeedHitSecond.layerNumber()
      ) return true;
  }
  return false;
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<LayerSpec> >& theLayersInSets,  const TrackerRecHit& theSeedHitSecond, const TrackerRecHit& theSeedHitThird) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0].subDet==seedingLayer.subDet && theLayersInSets[i][0].idLayer==seedingLayer.idLayer &&
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
      isOnDet =  seedingLayer.subDet==1;
      break;
      
    case 2: 
      //Pixel Disks
      isOnDet = seedingLayer.subDet==2;
      break;
      
    case 3:
      //Inner Barrel
      isOnDet = seedingLayer.subDet==3 && seedingLayer.idLayer < 4;
      break;
      
    case 4:
      //Inner Disks
      isOnDet = seedingLayer.subDet==4 && theRingNumber < 3;
      break;
      
    case 5:
      //Outer Barrel
      if(seedingAlgo == "TobTecLayerPairs"){
	isOnDet = seedingLayer.subDet==5 && seedingLayer.idLayer <3;
      }else {
	isOnDet = false;
      }
      break;
      
    case 6:
      //Tracker EndCap
      if(seedingAlgo == "PixelLessPairs"){
	isOnDet = seedingLayer.subDet==6 && seedingLayer.idLayer < 6 && theRingNumber < 3;
      }else if (seedingAlgo == "TobTecLayerPairs"){
	//	isOnDet = seedingLayer.subDet==6 && seedingLayer.idLayer < 8 && theRingNumber < 5;
	isOnDet = seedingLayer.subDet==6 && seedingLayer.idLayer < 8 && theRingNumber == 5;
      } else if (seedingAlgo == "MixedTriplets"){ 
	//	isOnDet = seedingLayer.subDet==6 && seedingLayer.idLayer == 2 && theRingNumber == 1;
	isOnDet = seedingLayer.subDet==6 && seedingLayer.idLayer < 4 && theRingNumber == 1;
      } else {
	isOnDet = seedingLayer.subDet==6;
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






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

bool
TrackerRecHit::makesAPairWith(const TrackerRecHit& anotherHit) const { 

  bool isAProperPair = false;

  unsigned int anotherSubDetId = anotherHit.subDetId();
  unsigned int anotherLayerNumber = anotherHit.layerNumber();
  isAProperPair = 
    // First hit on PXB1
    ( ( theSubDetId == 1 && theLayerNumber == 1 ) && (
      ( anotherSubDetId == 1 && anotherLayerNumber == 2) || 
      ( anotherSubDetId == 1 && anotherLayerNumber == 3) || 
      ( anotherSubDetId == 2 && anotherLayerNumber == 1) || 
      ( anotherSubDetId == 2 && anotherLayerNumber == 2) ) ) || 
    // First hit on PXB2
    ( ( theSubDetId == 1 && theLayerNumber == 2 ) && (
      ( anotherSubDetId == 1 && anotherLayerNumber == 3) || 
      ( anotherSubDetId == 2 && anotherLayerNumber == 1) || 
      ( anotherSubDetId == 2 && anotherLayerNumber == 2) ) ) ||
    // First Hit on PXD1
    ( ( theSubDetId == 2 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 2 && anotherLayerNumber == 2 ) ) ||
    // First Hit on PXD2
    ( ( theSubDetId == 2 && theLayerNumber == 2 ) && ( 
      ( anotherSubDetId == 6 && anotherLayerNumber == 1 ) ||
      ( anotherSubDetId == 6 && anotherLayerNumber == 2 ) ) ) ||
    // First Hit on TEC1
    ( ( theSubDetId == 6 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 2 ) ) ||
    // First Hit on TEC2
    ( ( theSubDetId == 6 && theLayerNumber == 2 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 3 ) ) ||

  //Pixelless Pairs  
   // First Hit on TIB1
    ( ( theSubDetId == 3 && theLayerNumber == 1 ) && 
      (( anotherSubDetId == 3 && anotherLayerNumber == 2 ) ||
       ( anotherSubDetId == 4 && anotherLayerNumber == 1 )) ) ||
    // First Hit on TID1
    ( ( theSubDetId == 4 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 4 && anotherLayerNumber == 2 ) ) ||
    // First Hit on TID2
    ( ( theSubDetId == 4 && theLayerNumber == 2 ) && 
      ( anotherSubDetId == 4 && anotherLayerNumber == 3 ) ) ||
    // First Hit on TID3
    ( ( theSubDetId == 4 && theLayerNumber == 3 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 1 ) ) ||
    // First Hit on TEC3
    ( ( theSubDetId == 6 && theLayerNumber == 3 ) && 
      (      ( anotherSubDetId == 6 && anotherLayerNumber == 4 ) || 
	     ( anotherSubDetId == 6 && anotherLayerNumber == 5 ))  ) ||
    // First Hit on TEC4
    ( ( theSubDetId == 6 && theLayerNumber == 4 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 5 ) ) ||

  //Tob-Tec pairs
  //first hit on TOB1 
    ( ( theSubDetId == 5 && theLayerNumber == 1 ) && 
      (( anotherSubDetId == 5 && anotherLayerNumber == 2 ) ||
       ( anotherSubDetId == 6 && anotherLayerNumber == 1 )) ) ||
    // First Hit on TEC1
    ( ( theSubDetId == 6 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 2 ) ) ||
    // First Hit on TEC2
    ( ( theSubDetId == 6 && theLayerNumber == 2 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 3 ) ) ||
    // First Hit on TEC3
    ( ( theSubDetId == 6 && theLayerNumber == 3 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 4 ) ) || 
      // ???     ( anotherSubDetId == 6 && anotherLayerNumber == 5 ) ) ||
    // First Hit on TEC4
    ( ( theSubDetId == 6 && theLayerNumber == 4 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 5 ) ) ||
    // First Hit on TEC5
    ( ( theSubDetId == 6 && theLayerNumber == 5 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 6 ) ) ||
    // First Hit on TEC6
    ( ( theSubDetId == 6 && theLayerNumber == 6 ) && 
      ( anotherSubDetId == 6 && anotherLayerNumber == 7 ) ) ;

  return isAProperPair;

} 

bool
TrackerRecHit::makesAPairWith3rd(const TrackerRecHit& anotherHit) const { 

  bool isAProperPair = false;

  unsigned int anotherSubDetId = anotherHit.subDetId();
  unsigned int anotherLayerNumber = anotherHit.layerNumber();
  isAProperPair = 
    // First hit on PXB1
    ( ( theSubDetId == 1 && theLayerNumber == 1 ) && (
      ( anotherSubDetId == 1 && anotherLayerNumber == 2) || 
      ( anotherSubDetId == 2 && anotherLayerNumber == 1) ) ) || 
      // First hit on PXB2
    ( ( theSubDetId == 1 && theLayerNumber == 2 ) && 
      ( anotherSubDetId == 1 && anotherLayerNumber == 3) ) || 
    // First Hit on PXD1
    ( ( theSubDetId == 2 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 2 && anotherLayerNumber == 2) ) ||
    // First Hit on PXD2
    ( ( theSubDetId == 2 && theLayerNumber == 2 ) &&  
      ( anotherSubDetId == 6 && anotherLayerNumber == 2 ) );

 return isAProperPair;

}
      
bool
TrackerRecHit::makesATripletWith(const TrackerRecHit& anotherHit,
				 const TrackerRecHit& yetAnotherHit ) const { 

  bool isAProperTriplet = false;

  unsigned int anotherSubDetId = anotherHit.subDetId();
  unsigned int anotherLayerNumber = anotherHit.layerNumber();
  unsigned int yetAnotherSubDetId = yetAnotherHit.subDetId();
  unsigned int yetAnotherLayerNumber = yetAnotherHit.layerNumber();
  isAProperTriplet = 
    // First hit on PXB1, second on PXB2
    ( ( theSubDetId == 1 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 1 && anotherLayerNumber == 2) && ( 
      ( yetAnotherSubDetId == 1 && yetAnotherLayerNumber == 3) || 
      ( yetAnotherSubDetId == 2 && yetAnotherLayerNumber == 1) || 
      ( yetAnotherSubDetId == 3 && yetAnotherLayerNumber == 1) ) ) || 
    // First hit on PXB1, second on PXB3 
    ( ( theSubDetId == 1 && theLayerNumber == 1 ) &&
      ( anotherSubDetId == 1 && anotherLayerNumber == 3) && 
      ( yetAnotherSubDetId == 3 && yetAnotherLayerNumber == 1) ) || 
    // First hit on PXB2, second on PXB3 
    ( ( theSubDetId == 1 && theLayerNumber == 2 ) &&
      ( anotherSubDetId == 1 && anotherLayerNumber == 3) && 
      ( yetAnotherSubDetId == 3 && yetAnotherLayerNumber == 1) ) || 
    // First Hit on PXB1, second on PXD1
    ( ( theSubDetId == 1 && theLayerNumber == 1 ) &&
      ( anotherSubDetId == 2 && anotherLayerNumber == 1) && ( 
      ( yetAnotherSubDetId == 2 && yetAnotherLayerNumber == 2) || 
      ( yetAnotherSubDetId == 4 && yetAnotherLayerNumber == 1) || 
      ( yetAnotherSubDetId == 4 && yetAnotherLayerNumber == 2) ) ) || 
    // First Hit on PXD1, second on PXD2
    ( ( theSubDetId == 2 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 2 && anotherLayerNumber == 2 ) && (
      ( yetAnotherSubDetId == 6 && yetAnotherLayerNumber == 1 ) ||
      ( yetAnotherSubDetId == 6 && yetAnotherLayerNumber == 2 ) ) ) ||
    // First hit on TIB1 (pixel less)
    ( ( theSubDetId == 3 && theLayerNumber == 1 ) && 
      ( anotherSubDetId == 3 && anotherLayerNumber == 2 ) && 
      ( yetAnotherSubDetId == 3 && yetAnotherLayerNumber == 3 ) );
  
  return isAProperTriplet;
  
}





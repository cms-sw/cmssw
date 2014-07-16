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
  int subDetId = theDetId.subdetId();
  theGeomDet = theGeometry->idToDet(theDetId);
  seedingLayer=LayerSpec::createFromDetId(theDetId,*tTopo);
  if ( subDetId == StripSubdetector::TIB) { 
     
    theCylinderNumber = TrackerInteractionGeometry::TIB+seedingLayer.layerNumber;
    forward = false;
  } else if (subDetId==  StripSubdetector::TOB ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::TOB+seedingLayer.layerNumber;
    forward = false;
  } else if ( subDetId ==  StripSubdetector::TID) { 
    
    theCylinderNumber = TrackerInteractionGeometry::TID+seedingLayer.layerNumber;
    theRingNumber = tTopo->tidRing(theDetId);
    forward = true;
  } else if ( subDetId ==  StripSubdetector::TEC ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::TEC+seedingLayer.layerNumber;
    theRingNumber = tTopo->tecRing(theDetId);
    forward = true;
  } else if ( subDetId ==  PixelSubdetector::PixelBarrel ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::PXB+seedingLayer.layerNumber;
    forward = false;
  } else if ( subDetId ==  PixelSubdetector::PixelEndcap ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::PXD+seedingLayer.layerNumber;
    forward = true;
  }
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<LayerSpec> >& theLayersInSets) const{ 
  
  for(unsigned int i=0; i<theLayersInSets.size(); ++i) {
    if(theLayersInSets[i][0]==seedingLayer) return true;
  }

  return false;
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<LayerSpec> >& theLayersInSets,  const TrackerRecHit& theSeedHitSecond) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0]==seedingLayer && 
        theLayersInSets[i][1]==theSeedHitSecond.getSeedingLayer()
        ) 
        return true;
  }
  return false;
}

bool
TrackerRecHit::isOnRequestedDet(const std::vector<std::vector<LayerSpec> >& theLayersInSets,  const TrackerRecHit& theSeedHitSecond, const TrackerRecHit& theSeedHitThird) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0]==seedingLayer && 
        theLayersInSets[i][1]==theSeedHitSecond.getSeedingLayer() &&
        theLayersInSets[i][2]==theSeedHitThird.getSeedingLayer()  
      ) return true;
  }
  return false;
}



#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

TrajectorySeedHitCandidate::TrajectorySeedHitCandidate(const FastTrackerRecHit * hit, 
			     const TrackerGeometry* theGeometry,
			     const TrackerTopology* tTopo) :
  theHit(hit),
  theRingNumber(0),
  theCylinderNumber(0),
  theLocalError(0.),
  theLargerError(0.)
     
{ 
  init(theGeometry, tTopo);
}

void
TrajectorySeedHitCandidate::init(const TrackerGeometry* theGeometry, const TrackerTopology *tTopo) { 

  const DetId& theDetId = hit()->geographicalId();
  int subDetId = theDetId.subdetId();
  theGeomDet = theGeometry->idToDet(theDetId);
  seedingLayer=TrackingLayer::createFromDetId(theDetId,*tTopo);
  if ( subDetId == StripSubdetector::TIB) { 
     
    theCylinderNumber = TrackerInteractionGeometry::TIB+seedingLayer.getLayerNumber();
    forward = false;
  } else if (subDetId==  StripSubdetector::TOB ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::TOB+seedingLayer.getLayerNumber();
    forward = false;
  } else if ( subDetId ==  StripSubdetector::TID) { 
    
    theCylinderNumber = TrackerInteractionGeometry::TID+seedingLayer.getLayerNumber();
    theRingNumber = tTopo->tidRing(theDetId);
    forward = true;
  } else if ( subDetId ==  StripSubdetector::TEC ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::TEC+seedingLayer.getLayerNumber();
    theRingNumber = tTopo->tecRing(theDetId);
    forward = true;
  } else if ( subDetId ==  PixelSubdetector::PixelBarrel ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::PXB+seedingLayer.getLayerNumber();
    forward = false;
  } else if ( subDetId ==  PixelSubdetector::PixelEndcap ) { 
     
    theCylinderNumber = TrackerInteractionGeometry::PXD+seedingLayer.getLayerNumber();
    forward = true;
  }
}

bool
TrajectorySeedHitCandidate::isOnRequestedDet(const std::vector<std::vector<TrackingLayer> >& theLayersInSets) const{ 
  
  for(unsigned int i=0; i<theLayersInSets.size(); ++i) {
    if(theLayersInSets[i][0]==seedingLayer) return true;
  }

  return false;
}

bool
TrajectorySeedHitCandidate::isOnRequestedDet(const std::vector<std::vector<TrackingLayer> >& theLayersInSets,  const TrajectorySeedHitCandidate& theSeedHitSecond) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0]==seedingLayer && 
        theLayersInSets[i][1]==theSeedHitSecond.getTrackingLayer()
        ) 
        return true;
  }
  return false;
}

bool
TrajectorySeedHitCandidate::isOnRequestedDet(const std::vector<std::vector<TrackingLayer> >& theLayersInSets,  const TrajectorySeedHitCandidate& theSeedHitSecond, const TrajectorySeedHitCandidate& theSeedHitThird) const{ 

  for(unsigned int i=0; i<theLayersInSets.size(); ++i){
    if( theLayersInSets[i][0]==seedingLayer && 
        theLayersInSets[i][1]==theSeedHitSecond.getTrackingLayer() &&
        theLayersInSets[i][2]==theSeedHitThird.getTrackingLayer()  
      ) return true;
  }
  return false;
}



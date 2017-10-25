#ifndef FastSimulation_Tracking_TrajectorySeedHitCandidate_H_

#define FastSimulation_Tracking_TrajectorySeedHitCandidate_H_

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastProjectedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastMatchedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


#include "FastSimulation/Tracking/interface/TrackingLayer.h"

#include <vector>

class TrackerTopology;

class TrajectorySeedHitCandidate
{
public:
  
  /// Default Constructor
  TrajectorySeedHitCandidate():
    theHit(nullptr),
    seedingLayer()
    
   {
   }

  /// Constructor from a FastTrackerRecHit and topology
  TrajectorySeedHitCandidate(const FastTrackerRecHit * hit, const TrackerTopology* tTopo);
  
  /// The Hit itself
  inline const FastTrackerRecHit * hit() const { return theHit; }
      
  inline const TrackingLayer& getTrackingLayer() const
  {
    return seedingLayer;
  }
  
  /// The subdet Id
  inline unsigned int subDetId() const { return seedingLayer.getSubDetNumber(); }
  
  /// The Layer Number
  inline unsigned int layerNumber() const { return seedingLayer.getLayerNumber(); }
  
  /// The local position
  inline LocalPoint localPosition() const { return hit()->localPosition(); }  
  /// Check if the hit is on one of the requested detector
  //  bool isOnRequestedDet(const std::vector<unsigned int>& whichDet) const;


  /// Check if two hits are on the same layer of the same subdetector
  inline bool isOnTheSameLayer(const TrajectorySeedHitCandidate& other) const {
    
    return seedingLayer==other.seedingLayer;
  }


  

 private:
  
  const FastTrackerRecHit * theHit;
  TrackingLayer seedingLayer;

};

#endif


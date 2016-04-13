#ifndef FastSimulation_Tracking_TrajectorySeedHitCandidate_H_

#define FastSimulation_Tracking_TrajectorySeedHitCandidate_H_

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
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

class TrajectorySeedHitCandidate {
public:
  
  /// Default Constructor
  TrajectorySeedHitCandidate() :
    theHit(0),
    seedingLayer(),
    theRingNumber(0), 
    theCylinderNumber(0), 
    forward(false)
    
   {
    
   }
    
  /// Soft Copy Constructor from private members
  /// lv: do we need this one?
  TrajectorySeedHitCandidate( const FastTrackerRecHit * hit, 
			      const TrajectorySeedHitCandidate& other ) : 
    theHit(hit),
    seedingLayer(other.getTrackingLayer()),
    theRingNumber(other.ringNumber()), 
    theCylinderNumber(other.cylinderNumber()), 
    forward(other.isForward())
    
    {
        
    }

  /// Constructor from a FastTrackerRecHit and topology
  TrajectorySeedHitCandidate(const FastTrackerRecHit * hit, 
			     const TrackerTopology* tTopo);
  
  /// Initialization at construction time
  void init(const TrackerTopology *tTopo);
  
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
  
  /// The Ring Number
  inline unsigned int ringNumber() const { return theRingNumber; }
  
  /// The global layer number in the nested cylinder geometry
  unsigned int cylinderNumber() const { return theCylinderNumber; }

  /// Is it a forward hit ?
  inline bool isForward() const { return forward; }

  /// The local position
  inline LocalPoint localPosition() const { return hit()->localPosition(); }  
  /// Check if the hit is on one of the requested detector
  //  bool isOnRequestedDet(const std::vector<unsigned int>& whichDet) const;

  /// request check with 1, 2 and 3 seeds
  bool isOnRequestedDet(const std::vector<std::vector<TrackingLayer> >& theLayersInSets) const;
  bool isOnRequestedDet(const std::vector<std::vector<TrackingLayer> >& theLayersInSets, const TrajectorySeedHitCandidate& theSeedHitSecond) const;
  bool isOnRequestedDet(const std::vector<std::vector<TrackingLayer> >& theLayersInSets, const TrajectorySeedHitCandidate& theSeedHitSecond, const TrajectorySeedHitCandidate& theSeedHitThird) const;


  /// Check if two hits are on the same layer of the same subdetector
  inline bool isOnTheSameLayer(const TrajectorySeedHitCandidate& other) const {
    
    return seedingLayer==other.seedingLayer;
  }

  inline bool operator!=(const TrajectorySeedHitCandidate& aHit) const {
    return 
      aHit.hit()->localPosition().x() != this->hit()->localPosition().x() ||
      aHit.hit()->localPosition().y() != this->hit()->localPosition().y() ||
      aHit.hit()->localPosition().z() != this->hit()->localPosition().z();
  }

  

 private:
  
  const FastTrackerRecHit * theHit;
  TrackingLayer seedingLayer;
  unsigned int theRingNumber;
  unsigned int theCylinderNumber;
  bool forward;

};

#endif


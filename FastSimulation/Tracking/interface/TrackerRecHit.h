#ifndef FastSimulation_Tracking_TrackerRecHit_H_
#define FastSimulation_Tracking_TrackerRecHit_H_

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <vector>

class TrackerTopology;

/** A class that gives some properties of the Tracker Layers in FAMOS
 */

class TrackerRecHit {
public:
  
  /// Default Constructor
  TrackerRecHit() :
    theSplitHit(0),
    theMatchedHit(0),
    theGeomDet(0),
    theSubDetId(0),
    theLayerNumber(0),
    theRingNumber(0), 
    theCylinderNumber(0), 
    theLocalError(0.),
    theLargerError(0.),
    forward(false) {}

  /// Soft Copy Constructor from private members
  TrackerRecHit( const SiTrackerGSRecHit2D* theSplitHit, 
		 const TrackerRecHit& other ) : 
    theSplitHit(theSplitHit),
    theMatchedHit(0),
    theGeomDet(other.geomDet()),
    theSubDetId(other.subDetId()),
    theLayerNumber(other.layerNumber()),
    theRingNumber(other.ringNumber()), 
    theCylinderNumber(other.cylinderNumber()), 
    theLocalError(0.),
    theLargerError(0.),
    forward(other.isForward()) {}

  /// Constructor from a GSRecHit and the Geometry
  TrackerRecHit(const SiTrackerGSRecHit2D* theHit, 
		const TrackerGeometry* theGeometry,
		const TrackerTopology* tTopo);
  
  TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
		const TrackerGeometry* theGeometry,
		const TrackerTopology *tTopo);

  /// Initialization at construction time
  void init(const TrackerGeometry* theGeometry,
	    const TrackerTopology *tTopo);
  
  // TrackerRecHit(const SiTrackerGSMatchedRecHit2D* theHit, 
  //		const TrackerGeometry* theGeometry);
  
  /// The Hit itself
  //  const SiTrackerGSRecHit2D* hit() const { return theHit; }
  inline const SiTrackerGSMatchedRecHit2D* matchedHit() const { return theMatchedHit; }
  inline const SiTrackerGSRecHit2D* splitHit() const { return theSplitHit; }
  
  inline const GSSiTrackerRecHit2DLocalPos* hit() const { 
    return theSplitHit ? (GSSiTrackerRecHit2DLocalPos*)theSplitHit : 
      (GSSiTrackerRecHit2DLocalPos*)theMatchedHit; }
  
  /// The subdet Id
  inline unsigned int subDetId() const { return theSubDetId; }
  
  /// The Layer Number
  inline unsigned int layerNumber() const { return theLayerNumber; }
  
  /// The Ring Number
  inline unsigned int ringNumber() const { return theRingNumber; }
  
  /// The global layer number in the nested cylinder geometry
  unsigned int cylinderNumber() const { return theCylinderNumber; }

  /// Is it a forward hit ?
  inline bool isForward() const { return forward; }

  /// The GeomDet
  inline const GeomDet* geomDet() const { return theGeomDet; }

  /// The global position
  inline GlobalPoint globalPosition() const { 
    return theGeomDet->surface().toGlobal(hit()->localPosition());
  }

  /// The local position
  inline LocalPoint localPosition() const { return hit()->localPosition(); }

  /// Check if the hit is on one of the requested detector
    //  bool isOnRequestedDet(const std::vector<unsigned int>& whichDet) const;
  bool isOnRequestedDet(const std::vector<unsigned int>& whichDet, const std::string& seedingAlgo) const; 
  bool isOnRequestedDet(const std::vector<std::string>& layerList) const; // AG

  /// Check if a pair is on the proper combination of detectors
  bool makesAPairWith(const TrackerRecHit& anotherHit) const;
  bool makesAPairWith3rd(const TrackerRecHit& anotherHit) const;

  /// Check if a triplet is on the proper combination of detectors
  bool makesATripletWith(const TrackerRecHit& anotherHit,
			 const TrackerRecHit& yetAnotherHit) const;

  /// Check if two hits are on the same layer of the same subdetector
  inline bool isOnTheSameLayer(const TrackerRecHit& other) const {
    
    return 
      theSubDetId == other.subDetId() && 
      theLayerNumber == other.layerNumber();
  }

  // The smaller local error
  double localError() { 

    // Check if it has been already computed
    if ( theLocalError != 0. ) return theLocalError;

    // Otherwise, compute it!
    double xx = hit()->localPositionError().xx();
    double yy = hit()->localPositionError().yy();
    double xy = hit()->localPositionError().xy();
    double delta = std::sqrt((xx-yy)*(xx-yy)+4.*xy*xy);
    theLocalError = 0.5 * (xx+yy-delta);
    return theLocalError;

  }
  
  // The larger local error
  double largerError() { 

    // Check if it has been already computed
    if ( theLargerError != 0. ) return theLargerError;

    // Otherwise, compute it!
    double xx = hit()->localPositionError().xx();
    double yy = hit()->localPositionError().yy();
    double xy = hit()->localPositionError().xy();
    double delta = std::sqrt((xx-yy)*(xx-yy)+4.*xy*xy);
    theLargerError = 0.5 * (xx+yy+delta);
    return theLargerError;

  }
  
  inline bool operator!=(const TrackerRecHit& aHit) const {
    return 
      aHit.geomDet() != this->geomDet() ||
      aHit.hit()->localPosition().x() != this->hit()->localPosition().x() ||
      aHit.hit()->localPosition().y() != this->hit()->localPosition().y() ||
      aHit.hit()->localPosition().z() != this->hit()->localPosition().z();
  }

 private:
  
  const SiTrackerGSRecHit2D* theSplitHit;
  const SiTrackerGSMatchedRecHit2D* theMatchedHit;
  const GeomDet* theGeomDet;
  unsigned int theSubDetId; 
  unsigned int theLayerNumber;
  unsigned int theRingNumber;
  unsigned int theCylinderNumber;
  double theLocalError;
  double theLargerError;
  bool forward;
  
};

#endif


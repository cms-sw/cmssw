#ifndef _VertexRecHit_H
#define _VertexRecHit_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h" 
#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"

/** A class that makes a Vertex appear as a RecHit, so it can be used
 *  in a track fit in the same way as other hits.
 *  Useful for adding a vertex constraint to a track.
 */

class VertexRecHit : RecHit2DLocalPos {
public:
  
  // constructor
  VertexRecHit(const LocalPoint & lp, 
	       const LocalError & le) :
     theLocalPosition(lp), theLocalPositionError(le) {}

  // access
  virtual LocalPoint localPosition() const { return theLocalPosition; }

  virtual LocalError localPositionError() const {
    return theLocalPositionError;
  } 

  virtual VertexRecHit * clone() const {return (VertexRecHit*)this; }

  virtual const TrackingRecHit* hit() const {return (const TrackingRecHit*)this;}

  virtual DetId geographicalId() const {return DetId(0);}


  virtual MeasurementPoint measurementPosition() const { 
    MeasurementPoint mp(theLocalPosition.x(), theLocalPosition.y());
    return mp;
  }

  virtual MeasurementError measurementError() const {
    MeasurementError me(theLocalPositionError.xx(), 
			theLocalPositionError.xy(), 
			theLocalPositionError.yy());
    return me;
  } 

private:
  LocalPoint theLocalPosition;
  LocalError theLocalPositionError;

};
#endif


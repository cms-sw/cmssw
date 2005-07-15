#ifndef _TRACKER_STRIP_TOPOLOGY_H_
#define _TRACKER_STRIP_TOPOLOGY_H_

#include "Geometry/CommonTopologies/interface/Topology.h"

/** Interface for all strip topologies.
 *  Extends the Topology interface with methods relevant for
 *  strip or wire detectors.
 */

class StripTopology : public Topology {
public:

  virtual ~StripTopology() {}

  virtual LocalPoint localPosition( const MeasurementPoint& ) const = 0;
  virtual LocalPoint localPosition( float strip ) const = 0;
  virtual LocalError localError( float strip, float stripErr2 ) const = 0;
  virtual LocalError localError( const MeasurementPoint&,
                                 const MeasurementError& ) const = 0;
  virtual float strip( const LocalPoint& ) const = 0;
  virtual float pitch() const = 0;
  virtual float localPitch( const LocalPoint& ) const = 0; 
  virtual float stripAngle( float strip ) const = 0;
  virtual int nstrips() const = 0;

  virtual float stripLength() const = 0;
  virtual float localStripLength(const LocalPoint& aLP) const = 0;

};

#endif

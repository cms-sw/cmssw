#ifndef Geometry_CommonTopologies_StripTopology_H
#define Geometry_CommonTopologies_StripTopology_H

#include "Geometry/CommonTopologies/interface/Topology.h"

/** Interface for all strip topologies.
 *  Extends the Topology interface with methods relevant for
 *  strip or wire detectors.
 */

class StripTopology : public Topology {
public:

  virtual ~StripTopology() {}

  // GF: I hate the stupid hiding feature of C++, see
  // http://www.parashift.com/c%2B%2B-faq-lite/strange-inheritance.html#faq-23.9
  using Topology::localPosition;
  virtual LocalPoint localPosition( float strip ) const = 0;
  /// conversion taking also the angle from the predicted track state 
  virtual LocalPoint localPosition( float strip, const Topology::LocalTrackAngles &dir ) const { 
    return localPosition(strip); 
  }
  virtual LocalError localError( float strip, float stripErr2 ) const = 0;

  /// conversion taking also the angle from the predicted track state
  virtual LocalError localError( float strip, float stripErr2, const Topology::LocalTrackAngles &dir ) const {
    return localError(strip, stripErr2); 
  } 
  virtual LocalError localError( const MeasurementPoint&,
                                 const MeasurementError& ) const = 0;

  /// conversion taking also the angle from the predicted track state
  virtual LocalError localError( const MeasurementPoint& mp,
                                 const MeasurementError& me,
                                 const Topology::LocalTrackAngles &dir ) const {
    return localError(mp, me);
  }
  virtual float strip( const LocalPoint& ) const = 0;

  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float strip( const LocalPoint& lp, const Topology::LocalTrackAngles &ltp ) const {
    return strip(lp);
  }
  virtual float pitch() const = 0;
  virtual float localPitch( const LocalPoint& ) const = 0; 

  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float localPitch( const LocalPoint& lp, const Topology::LocalTrackAngles &ltp ) const {
    return localPitch(lp);
  }
  virtual float stripAngle( float strip ) const = 0;

  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float stripAngle( float strip, const Topology::LocalTrackAngles &dir ) const {
    return stripAngle( strip );
  }
  virtual int nstrips() const = 0;

  virtual float stripLength() const = 0;
  virtual float localStripLength(const LocalPoint& aLP) const = 0;

  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float localStripLength( const LocalPoint& lp, const Topology::LocalTrackAngles &ltp ) const {
    return localStripLength(lp);
  }

};

#endif

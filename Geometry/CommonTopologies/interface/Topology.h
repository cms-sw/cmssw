#ifndef Geometry_CommonTopologies_Topology_H
#define Geometry_CommonTopologies_Topology_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/LocalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"

class GeomDetType;

/** Abstract component defining the conversion between the local frame of
 *  a detector and the frame defined by the readout channels ,
 *  the so called measurement frame. For example, in a strip detector
 *  the strips define a coordinate frame (from 0 to Nstrips in one direction
 *  and from 0 to 1 in the other), and each local point can be mapped to a point 
 *  in this frame. The mapping may be non-linear (for example for trapezoidal 
 *  strips). 
 *
 *  The topology should be the ONLY place where this mapping is defined.
 *  The Digitizer uses the Topology to transform energy deposits in the 
 *  local frame into signals on the readout channels, and the clusterizer
 *  (or the RecHit) uses the Topology for the inverse transformation,
 *  from channel numbers to local coordinates. 
 */

class Topology {
public:
    /** Track angles in the local frame, needed to handle surface deformations */
    class LocalTrackAngles {
        public:
            LocalTrackAngles(double dxdz, double dydz) : dxdz_(dxdz), dydz_(dydz) {}
            double dydz() const { return dydz_; }
            double dxdz() const { return dxdz_; }
        private:
            double dxdz_, dydz_;
    };


  virtual ~Topology() {}
  
  // Conversion between measurement (strip, pixel, ...) coordinates
  // and local cartesian coordinates

  virtual LocalPoint localPosition( const MeasurementPoint& ) const = 0;

  virtual LocalError 
  localError( const MeasurementPoint&, const MeasurementError& ) const = 0;

  virtual MeasurementPoint measurementPosition( const LocalPoint&) const = 0;

  virtual MeasurementError 
  measurementError( const LocalPoint&, const LocalError& ) const = 0;

  virtual int channel( const LocalPoint& p) const = 0;

  // new sets of methods taking also an angle
  /// conversion taking also the angle from the predicted track state 
  virtual LocalPoint localPosition( const MeasurementPoint &mp, const LocalTrackAngles &dir) const {
      return localPosition(mp);
  }

  /// conversion taking also the angle from the predicted track state 
  virtual LocalError 
  localError( const MeasurementPoint &mp, const MeasurementError &me, const LocalTrackAngles &dir ) const {
      return localError(mp,me);
  }

  /// conversion taking also the angle from the track state
  virtual MeasurementPoint measurementPosition( const LocalPoint &lp, const LocalTrackAngles &dir) const {
      return measurementPosition(lp);
  }

  /// conversion taking also the angle from the track state
  virtual MeasurementError 
  measurementError( const LocalPoint &lp, const LocalError &le, const LocalTrackAngles &dir) const {
      return measurementError(lp, le);
  }

  /// conversion taking also the angle from the track state
  virtual int channel( const LocalPoint &lp, const LocalTrackAngles &dir) const {
      return channel(lp);
  }


private:

};

#endif

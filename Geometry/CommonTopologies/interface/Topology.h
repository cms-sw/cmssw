#ifndef Geometry_CommonTopologies_Topology_H
#define Geometry_CommonTopologies_Topology_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

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
 *
 *  If the surface of a Topology deviates from its perfect shape,
 *  e.g. a bowed instead of a flat surface (bowed silicon sensors)
 *  or if built from two surfaces that might be misaligned with respect
 *  to each other (double sensor silicon modules), conversion might depend 
 *  on a track prediction.
 *  Conversions from local to measurement frame therefore need the
 *  'LocalTrackAngles', conversions the other way round need also track
 *   position predictions, bound together with the angles to a 'LocalTrackPred'.
 *  Default implementation of methods requiring more arguments is to
 *  call the simple method, ignoring the track state.
 *  Concrete implementations should overwrite this where appropiate.
 */

class Topology {
public:
  typedef Basic2DVector<double> Vector2D;
  typedef Vector2D::MathVector MathVector2D;
  /** Track angles in the local frame, needed to handle surface deformations */
  class LocalTrackAngles : public Vector2D {
  public:
    typedef Basic2DVector<double> Base;
    LocalTrackAngles() {}
    LocalTrackAngles(const Base &v) : Base(v) {}
    LocalTrackAngles(double dxdz, double dydz) : Base(dxdz, dydz) {}
    double dxdz() const { return x(); }
    double dydz() const { return y(); }
  };
  typedef Point2DBase<double, LocalTag> Local2DPoint;
  /** Track prediction in local frame (2D point and angles), 
	needed to handle surface deformations*/
  class LocalTrackPred {
  public:
    LocalTrackPred() {}
    LocalTrackPred(double x, double y, double dxdz, double dydz) : point_(x, y), angles_(dxdz, dydz) {}
    /// Ctr. from local trajectory parameters as AlgebraicVector5 (q/p, dxdz, dydz, x, y)
    /// e.g. from 'LocalTrajectoryParameters::vector()'
    LocalTrackPred(const AlgebraicVector5 &localTrajPar)
        : point_(localTrajPar[3], localTrajPar[4]), angles_(localTrajPar[1], localTrajPar[2]) {}
    const Local2DPoint &point() const { return point_; }
    const LocalTrackAngles &angles() const { return angles_; }

  private:
    Local2DPoint point_;       /// local x, y
    LocalTrackAngles angles_;  /// local dxdz, dydz
  };

  virtual ~Topology() {}

  // Conversion between measurement (strip, pixel, ...) coordinates
  // and local cartesian coordinates

  virtual LocalPoint localPosition(const MeasurementPoint &) const = 0;

  virtual LocalError localError(const MeasurementPoint &, const MeasurementError &) const = 0;

  virtual MeasurementPoint measurementPosition(const LocalPoint &) const = 0;

  virtual MeasurementError measurementError(const LocalPoint &, const LocalError &) const = 0;

  virtual int channel(const LocalPoint &p) const = 0;

  // new sets of methods taking also an angle
  /// conversion taking also the angle from the predicted track state
  virtual LocalPoint localPosition(const MeasurementPoint &mp, const LocalTrackPred & /*trkPred*/) const {
    return localPosition(mp);
  }

  /// conversion taking also the angle from the predicted track state
  virtual LocalError localError(const MeasurementPoint &mp,
                                const MeasurementError &me,
                                const LocalTrackPred & /*trkPred*/) const {
    return localError(mp, me);
  }

  /// conversion taking also the angle from the track state
  virtual MeasurementPoint measurementPosition(const LocalPoint &lp, const LocalTrackAngles & /*dir*/) const {
    return measurementPosition(lp);
  }

  /// conversion taking also the angle from the track state
  virtual MeasurementError measurementError(const LocalPoint &lp,
                                            const LocalError &le,
                                            const LocalTrackAngles & /*dir*/) const {
    return measurementError(lp, le);
  }

  /// conversion taking also the angle from the track state
  virtual int channel(const LocalPoint &lp, const LocalTrackAngles & /*dir*/) const { return channel(lp); }

private:
};

#endif

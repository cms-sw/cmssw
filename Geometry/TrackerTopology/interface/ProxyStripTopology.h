#ifndef Geometry_TrackerTopology_ProxyStripTopology_H
#define Geometry_TrackerTopology_ProxyStripTopology_H

/// ProxyStripTopology
///
/// Class derived from StripTopology that serves as a proxy to the
/// actual topology for a given StripGeomDetType. In addition the
/// class holds a pointer to the surface deformation parameters.
/// ProxyStripTopology takes over ownership of the surface
/// deformation parameters.
/// All inherited pure virtual methods that take the angle from the
/// predicted track state as a parameter are reimplemented in order
/// to apply corrections due to the surface deformations.
///
///  \author    : Andreas Mussgiller
///  date       : November 2010
///  $Revision: 1.2 $
///  $Date: 2010/11/17 15:55:09 $
///  (last update by $Author: flucke $)

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

class StripGeomDetType;
class BoundPlane;
class Bounds;

class ProxyStripTopology : public StripTopology {
public:

  ProxyStripTopology(StripGeomDetType* type, BoundPlane * bp);

  virtual LocalPoint localPosition( const MeasurementPoint& mp ) const;
  /// conversion taking also the angle from the predicted track state 
  virtual LocalPoint localPosition( const MeasurementPoint& mp, const Topology::LocalTrackAngles &dir ) const;

  virtual LocalPoint localPosition( float strip ) const;
  /// conversion taking also the angle from the predicted track state
  virtual LocalPoint localPosition( float strip, const Topology::LocalTrackAngles &dir) const;

  virtual LocalError localError( float strip, float stripErr2 ) const;
  /// conversion taking also the angle from the predicted track state
  virtual LocalError localError( float strip, float stripErr2, const Topology::LocalTrackAngles &dir ) const;

  virtual LocalError localError( const MeasurementPoint& mp,
				 const MeasurementError& me) const;
  /// conversion taking also the angle from the predicted track state
  virtual LocalError localError( const MeasurementPoint& mp,
				 const MeasurementError& me,
				 const Topology::LocalTrackAngles &dir) const;
  
  virtual MeasurementPoint measurementPosition( const LocalPoint& lp) const;
  virtual MeasurementPoint measurementPosition( const LocalPoint &lp, 
						const Topology::LocalTrackAngles &dir) const;

  virtual MeasurementError measurementError( const LocalPoint& lp,
					     const LocalError& le ) const;
  virtual MeasurementError measurementError( const LocalPoint &lp, const LocalError &le,
					     const Topology::LocalTrackAngles &dir) const;
  
  virtual int channel( const LocalPoint& lp) const;
  virtual int channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir) const;
  
  virtual float strip( const LocalPoint& lp) const;
  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float strip( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const;

  virtual float pitch() const;
  virtual float localPitch( const LocalPoint& lp) const;
  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float localPitch( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const;
  
  virtual float stripAngle( float strip ) const;
  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float stripAngle( float strip, const Topology::LocalTrackAngles &dir ) const;

  virtual int nstrips() const;
  
  virtual float stripLength() const;
  virtual float localStripLength(const LocalPoint& lp) const;
  virtual float localStripLength( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const;
  
  virtual const GeomDetType& type() const;
  virtual StripGeomDetType& specificType() const;

  const SurfaceDeformation * surfaceDeformation() const {
    return theSurfaceDeformation.operator->();
  }
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

private:
  
  virtual const StripTopology& specificTopology() const;

  StripGeomDetType* theType;
  const Bounds & theBounds;
  DeepCopyPointerByClone<const SurfaceDeformation> theSurfaceDeformation;
};

#endif

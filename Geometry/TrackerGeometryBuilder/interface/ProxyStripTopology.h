#ifndef Geometry_TrackerTopology_ProxyStripTopology_H
#define Geometry_TrackerTopology_ProxyStripTopology_H

/// ProxyStripTopology
///
/// Class derived from StripTopology that serves as a proxy to the
/// actual topology for a given StripGeomDetType. In addition, the
/// class holds a pointer to the surface deformation parameters.
/// ProxyStripTopology takes over ownership of the surface
/// deformation parameters.
///
/// All inherited virtual methods that take the 
/// predicted track state as a parameter are reimplemented in order
/// to apply corrections due to the surface deformations.
//
/// The 'old' methods without the track predictions simply call
/// the method of the actual StripTopology.
/// While one could easily deduce corrections from the given
/// LocalPosition (and track angles 0) when converting from local frame
/// to measurement frame, this is not done to be consistent with the
/// methods converting the other way round where the essential y-coordinate
/// is basically missing (it is a 1D strip detector...)
///
///  \author    : Andreas Mussgiller
///  date       : November 2010
///  $Revision: 1.4 $
///  $Date: 2012/04/26 17:28:55 $
///  (last update by $Author: innocent $)

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
class BoundPlane;

class ProxyStripTopology GCC11_FINAL : public StripTopology {
public:

  ProxyStripTopology(StripGeomDetType* type, BoundPlane * bp);

  virtual LocalPoint localPosition( const MeasurementPoint& mp ) const { return specificTopology().localPosition(mp);}
  /// conversion taking also the predicted track state 
  virtual LocalPoint localPosition( const MeasurementPoint& mp, const Topology::LocalTrackPred &trkPred ) const;

  virtual LocalPoint localPosition( float strip ) const {return specificTopology().localPosition(strip);}
  /// conversion taking also the predicted track state
  virtual LocalPoint localPosition( float strip, const Topology::LocalTrackPred &trkPred) const;

  virtual LocalError localError( float strip, float stripErr2 ) const {return specificTopology().localError(strip, stripErr2);}
  /// conversion taking also the predicted track state
  virtual LocalError localError( float strip, float stripErr2, const Topology::LocalTrackPred &trkPred) const;

  virtual LocalError localError( const MeasurementPoint& mp,
				 const MeasurementError& me) const { return specificTopology().localError(mp, me);}
  /// conversion taking also the predicted track state
  virtual LocalError localError( const MeasurementPoint& mp,
				 const MeasurementError& me,
				 const Topology::LocalTrackPred &trkPred) const;
  
  virtual MeasurementPoint measurementPosition( const LocalPoint& lp) const  {return specificTopology().measurementPosition(lp);}
  virtual MeasurementPoint measurementPosition( const LocalPoint &lp, 
						const Topology::LocalTrackAngles &dir) const;

  virtual MeasurementError measurementError( const LocalPoint& lp,
					     const LocalError& le ) const { return specificTopology().measurementError(lp, le); }
  virtual MeasurementError measurementError( const LocalPoint &lp, const LocalError &le,
					     const Topology::LocalTrackAngles &dir) const;
  
  virtual int channel( const LocalPoint& lp) const {return specificTopology().channel(lp);}
  virtual int channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir) const;
  
  virtual float strip( const LocalPoint& lp) const { return specificTopology().strip(lp);}
  /// conversion taking also the track state (LocalTrajectoryParameters)
  virtual float strip( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const;

  virtual float pitch() const { return specificTopology().pitch(); }
  virtual float localPitch( const LocalPoint& lp) const { return specificTopology().localPitch(lp);}
  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  virtual float localPitch( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const;
  
  virtual float stripAngle( float strip ) const { return specificTopology().stripAngle(strip);}

  virtual int nstrips() const {return specificTopology().nstrips();}
  
  virtual float stripLength() const {return specificTopology().stripLength();}
  virtual float localStripLength(const LocalPoint& lp) const { return specificTopology().localStripLength(lp);}
  virtual float localStripLength( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const;
  
  virtual const GeomDetType& type() const  { return *theType;}
  virtual StripGeomDetType& specificType() const  { return *theType;}

  const SurfaceDeformation * surfaceDeformation() const {
    return theSurfaceDeformation.operator->();
  }
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

  
  virtual const StripTopology& specificTopology() const {return specificType().specificTopology();}

private:

  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector
    positionCorrection(const LocalPoint &pos, const Topology::LocalTrackAngles &dir) const;
  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector
    positionCorrection(const Topology::LocalTrackPred &trk) const;

  StripGeomDetType* theType;
  float theLength, theWidth;
  DeepCopyPointerByClone<const SurfaceDeformation> theSurfaceDeformation;
};

#endif

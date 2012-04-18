#ifndef Geometry_TrackerTopology_ProxyPixelTopology_H
#define Geometry_TrackerTopology_ProxyPixelTopology_H

/// ProxyStripTopology
///
/// Class derived from PixelTopology that serves as a proxy to the
/// actual topology for a given PixelGeomDetType. In addition, the
/// class holds a pointer to the surface deformation parameters.
/// ProxyPixelTopology takes over ownership of the surface
/// deformation parameters.
///
/// All inherited virtual methods that take the predicted track
/// state as a parameter are reimplemented in order to apply
/// corrections due to the surface deformations.
//
/// The 'old' methods without the track predictions simply call
/// the method of the actual topology.
///
///  \author    : Andreas Mussgiller
///  date       : December 2010
///  $Revision: 1.3 $
///  $Date: 2011/02/11 11:16:28 $
///  (last update by $Author: flucke $)

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

class PixelGeomDetType;
class BoundPlane;

class ProxyPixelTopology GCC11_FINAL : public PixelTopology {
public:

  ProxyPixelTopology( PixelGeomDetType* type, BoundPlane * bp );

  virtual LocalPoint localPosition( const MeasurementPoint& ) const;
  /// conversion taking also the predicted track state 
  virtual LocalPoint localPosition( const MeasurementPoint& mp,
				    const Topology::LocalTrackPred &trkPred ) const;
  
  virtual LocalError localError( const MeasurementPoint&,
                                 const MeasurementError& ) const;
  /// conversion taking also the predicted track state
  virtual LocalError localError( const MeasurementPoint& mp,
				 const MeasurementError& me,
				 const Topology::LocalTrackPred &trkPred ) const;

  virtual MeasurementPoint measurementPosition( const LocalPoint & ) const;
  virtual MeasurementPoint measurementPosition( const LocalPoint &lp, 
						const Topology::LocalTrackAngles &dir ) const;

  virtual MeasurementError measurementError( const LocalPoint &lp, const LocalError &le ) const;
  virtual MeasurementError measurementError( const LocalPoint &lp, const LocalError &le,
					     const Topology::LocalTrackAngles &dir ) const;

  virtual int channel( const LocalPoint& ) const;
  virtual int channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir ) const;
  
  virtual std::pair<float,float> pixel( const LocalPoint& p) const;
  /// conversion taking also the angle from the track state
  virtual std::pair<float,float> pixel( const LocalPoint& p,
					const Topology::LocalTrackAngles &ltp ) const; 
  
  virtual std::pair<float,float> pitch() const;
  virtual int nrows() const;
  virtual int ncolumns() const;

  virtual float localX( const float mpX ) const;
  virtual float localX( const float mpX, const Topology::LocalTrackPred &trkPred ) const;
  virtual float localY( const float mpY ) const;
  virtual float localY( const float mpY, const Topology::LocalTrackPred &trkPred ) const;

  virtual bool isItBigPixelInX(const int ixbin) const {
    return specificTopology().isItBigPixelInX(ixbin);
  }
  virtual bool isItBigPixelInY(const int iybin) const {
    return specificTopology().isItBigPixelInY(iybin);
  }
  virtual bool containsBigPixelInX(const int& ixmin, const int& ixmax) const {
    return specificTopology().containsBigPixelInX(ixmin, ixmax);
  }
  virtual bool containsBigPixelInY(const int& iymin, const int& iymax) const {
    return specificTopology().containsBigPixelInY(iymin, iymax);
  }

  virtual bool isItEdgePixelInX(int ixbin) const {
    return specificTopology().isItEdgePixelInX(ixbin);
  }
  virtual bool isItEdgePixelInY(int iybin) const {
    return specificTopology().isItEdgePixelInY(iybin);
  }
  virtual bool isItEdgePixel(int ixbin, int iybin) const {
    return specificTopology().isItEdgePixel(ixbin, iybin);
  }

  virtual const GeomDetType& type() const;
  virtual PixelGeomDetType& specificType() const;

  const SurfaceDeformation * surfaceDeformation() const { 
    return theSurfaceDeformation.operator->();
  }
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

private:
  
  virtual const PixelTopology& specificTopology() const;
  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector
    positionCorrection(const LocalPoint &pos, const Topology::LocalTrackAngles &dir) const;
  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector
    positionCorrection(const Topology::LocalTrackPred &trk) const;
  
  PixelGeomDetType* theType;  
  float theLength, theWidth;
  DeepCopyPointerByClone<const SurfaceDeformation> theSurfaceDeformation;
};

#endif

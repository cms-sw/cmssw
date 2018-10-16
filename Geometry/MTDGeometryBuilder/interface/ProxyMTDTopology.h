#ifndef Geometry_MTDGeometryBuilder_ProxyMTDTopology_H
#define Geometry_MTDGeometryBuilder_ProxyMTDTopology_H

/// ProxyMTDTopology
///
/// Class derived from PixelTopology that serves as a proxy to the
/// actual topology for a given PixelGeomDetType. In addition, the
/// class holds a pointer to the surface deformation parameters.
/// ProxyMTDTopology takes over ownership of the surface
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

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"

class Plane;

class ProxyMTDTopology final : public PixelTopology {
public:

  ProxyMTDTopology( MTDGeomDetType const * type, Plane * bp );

  LocalPoint localPosition( const MeasurementPoint& ) const override;
  /// conversion taking also the predicted track state 
  LocalPoint localPosition( const MeasurementPoint& mp,
			    const Topology::LocalTrackPred &trkPred ) const override;
  
  LocalError localError( const MeasurementPoint&,
                                 const MeasurementError& ) const override;
  /// conversion taking also the predicted track state
  LocalError localError( const MeasurementPoint& mp,
				 const MeasurementError& me,
				 const Topology::LocalTrackPred &trkPred ) const override;

  MeasurementPoint measurementPosition( const LocalPoint & ) const override;
  MeasurementPoint measurementPosition( const LocalPoint &lp, 
						const Topology::LocalTrackAngles &dir ) const override;

  MeasurementError measurementError( const LocalPoint &lp, const LocalError &le ) const override;
  MeasurementError measurementError( const LocalPoint &lp, const LocalError &le,
					     const Topology::LocalTrackAngles &dir ) const override;

  int channel( const LocalPoint& ) const override;
  int channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir ) const override;
  
  std::pair<float,float> pixel( const LocalPoint& p) const override;
  /// conversion taking also the angle from the track state
  std::pair<float,float> pixel( const LocalPoint& p,
					const Topology::LocalTrackAngles &ltp ) const override; 
  
  std::pair<float,float> pitch() const override { return specificTopology().pitch(); }
  int nrows() const override { return specificTopology().nrows(); }
  int ncolumns() const override { return specificTopology().ncolumns(); }
  int rocsY() const override { return specificTopology().rocsY(); } 	 
  int rocsX() const override { return specificTopology().rocsX(); } 	 
  int rowsperroc() const override { return specificTopology().rowsperroc(); } 	 
  int colsperroc() const override { return specificTopology().colsperroc(); }
  float localX( const float mpX ) const override;
  float localX( const float mpX, const Topology::LocalTrackPred &trkPred ) const override;
  float localY( const float mpY ) const override;
  float localY( const float mpY, const Topology::LocalTrackPred &trkPred ) const override;

  bool isItBigPixelInX(const int ixbin) const override {
    return specificTopology().isItBigPixelInX(ixbin);
  }
  bool isItBigPixelInY(const int iybin) const override {
    return specificTopology().isItBigPixelInY(iybin);
  }
  bool containsBigPixelInX(int ixmin, int ixmax) const override {
    return specificTopology().containsBigPixelInX(ixmin, ixmax);
  }
  bool containsBigPixelInY(int iymin, int iymax) const override {
    return specificTopology().containsBigPixelInY(iymin, iymax);
  }

  bool isItEdgePixelInX(int ixbin) const override {
    return specificTopology().isItEdgePixelInX(ixbin);
  }
  bool isItEdgePixelInY(int iybin) const override {
    return specificTopology().isItEdgePixelInY(iybin);
  }
  bool isItEdgePixel(int ixbin, int iybin) const override {
    return specificTopology().isItEdgePixel(ixbin, iybin);
  }

  virtual const GeomDetType& type() const { return *theType;}

  virtual MTDGeomDetType const & specificType() const { return *theType; }

  const SurfaceDeformation * surfaceDeformation() const { 
    return theSurfaceDeformation.operator->();
  }
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);

  
  virtual const PixelTopology& specificTopology() const { return specificType().specificTopology(); }

private:

  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector
    positionCorrection(const LocalPoint &pos, const Topology::LocalTrackAngles &dir) const;
  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector
    positionCorrection(const Topology::LocalTrackPred &trk) const;
  
  MTDGeomDetType const * theType;  
  float theLength, theWidth;
  std::unique_ptr<const SurfaceDeformation> theSurfaceDeformation;
};

#endif

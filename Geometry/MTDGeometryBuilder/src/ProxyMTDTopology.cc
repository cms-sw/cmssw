#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"

////////////////////////////////////////////////////////////////////////////////
ProxyMTDTopology::ProxyMTDTopology(MTDGeomDetType const * type, BoundPlane * bp)
  :theType(type), theLength(bp->bounds().length()), theWidth(bp->bounds().width())
{
  
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyMTDTopology::localPosition( const MeasurementPoint& mp ) const
{
  return specificTopology().localPosition(mp);
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyMTDTopology::localPosition( const MeasurementPoint& mp, 
					    const Topology::LocalTrackPred &trkPred ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPosition(mp);
  
  // add correction from SurfaceDeformation
  const LocalPoint posOld(specificTopology().localPosition(mp)); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  
  return LocalPoint(posOld.x()+corr.x(), posOld.y()+corr.y(), posOld.z());
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyMTDTopology::localError( const MeasurementPoint& mp,
					 const MeasurementError& me ) const
{
  return specificTopology().localError(mp, me);
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyMTDTopology::localError( const MeasurementPoint& mp,
					 const MeasurementError& me, 
					 const Topology::LocalTrackPred &trkPred ) const
{
  // The topology knows to calculate the cartesian error from measurement frame.
  // But assuming no uncertainty on the SurfaceDeformation variables,
  // the errors do not change from a simple shift to compensate
  // that the track 'sees' the surface at another place than it thinks...
  return specificTopology().localError(mp, me);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementPoint ProxyMTDTopology::measurementPosition( const LocalPoint& lp ) const
{
  return specificTopology().measurementPosition(lp);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementPoint ProxyMTDTopology::measurementPosition( const LocalPoint& lp, 
							const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementPosition(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementPosition(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementError ProxyMTDTopology::measurementError( const LocalPoint &lp, const LocalError &le ) const
{
  return specificTopology().measurementError(lp, le);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementError ProxyMTDTopology::measurementError( const LocalPoint &lp, const LocalError &le,
						     const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementError(lp, le);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementError(posOrig, le);
}

////////////////////////////////////////////////////////////////////////////////
int ProxyMTDTopology::channel( const LocalPoint& lp) const
{
  return specificTopology().channel(lp);
}

////////////////////////////////////////////////////////////////////////////////
int ProxyMTDTopology::channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir) const
{
   if (!this->surfaceDeformation()) return specificTopology().channel(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());
    
  return specificTopology().channel(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
std::pair<float,float> ProxyMTDTopology::pixel( const LocalPoint& lp ) const
{
  return specificTopology().pixel(lp);
}

////////////////////////////////////////////////////////////////////////////////
std::pair<float,float> ProxyMTDTopology::pixel( const LocalPoint& lp,
						  const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().pixel(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());
  
  return specificTopology().pixel(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyMTDTopology::localX(const float mpX) const
{
  return specificTopology().localX(mpX);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyMTDTopology::localX(const float mpX,
			       const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localX(mpX);
  
  // add correction from SurfaceDeformation
  float xOld = specificTopology().localX(mpX); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  
  return xOld + corr.x();
}

////////////////////////////////////////////////////////////////////////////////
float ProxyMTDTopology::localY(const float mpY) const
{
  return specificTopology().localY(mpY);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyMTDTopology::localY(const float mpY,
			       const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localY(mpY);

  // add correction from SurfaceDeformation
  float yOld = specificTopology().localY(mpY); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  
  return yOld + corr.y();
}



////////////////////////////////////////////////////////////////////////////////
void ProxyMTDTopology::setSurfaceDeformation(const SurfaceDeformation * deformation)
{ 
  theSurfaceDeformation.reset(deformation);
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyMTDTopology::positionCorrection(const LocalPoint &pos,
				     const Topology::LocalTrackAngles &dir) const
{
  const SurfaceDeformation::Local2DPoint pos2D(pos.x(), pos.y());// change precision and dimension

  return this->surfaceDeformation()->positionCorrection(pos2D, dir,
							theLength, theWidth);
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyMTDTopology::positionCorrection(const Topology::LocalTrackPred &trk) const
{
  return this->surfaceDeformation()->positionCorrection(trk.point(), trk.angles(),
							theLength, theWidth);
}

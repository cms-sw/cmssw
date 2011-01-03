#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "Geometry/TrackerGeometryBuilder/interface/ProxyPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

////////////////////////////////////////////////////////////////////////////////
ProxyPixelTopology::ProxyPixelTopology(PixelGeomDetType* type, BoundPlane * bp)
  :theType(type), theBounds(bp->bounds())
{
  
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyPixelTopology::localPosition( const MeasurementPoint& mp ) const
{
  return this->localPosition(mp, Topology::LocalTrackPred(0., 0., 0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyPixelTopology::localPosition( const MeasurementPoint& mp, 
					      const Topology::LocalTrackPred &trkPred ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPosition(mp);
  
  // add correction from SurfaceDeformation
  const LocalPoint posOld(specificTopology().localPosition(mp)); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  
  return LocalPoint(posOld.x()+corr.x(), posOld.y()+corr.y(), posOld.z());
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyPixelTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me ) const
{
  return this->localError(mp, me, Topology::LocalTrackPred(0., 0., 0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyPixelTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me, 
					   const Topology::LocalTrackPred &trkPred ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localError(mp, me);

  // FIXME: Add code to actually use SurfaceDeformation

  return specificTopology().localError(mp, me);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementPoint ProxyPixelTopology::measurementPosition( const LocalPoint& lp ) const
{
  return this->measurementPosition(lp, Topology::LocalTrackAngles(0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
MeasurementPoint ProxyPixelTopology::measurementPosition( const LocalPoint& lp, 
							  const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementPosition(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementPosition(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementError ProxyPixelTopology::measurementError( const LocalPoint &lp, const LocalError &le ) const
{
  return this->measurementError(lp, le, Topology::LocalTrackAngles(0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
MeasurementError ProxyPixelTopology::measurementError( const LocalPoint &lp, const LocalError &le,
						       const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementError(lp, le);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementError(posOrig, le);
}

////////////////////////////////////////////////////////////////////////////////
int ProxyPixelTopology::channel( const LocalPoint& lp) const
{
  return this->channel(lp, Topology::LocalTrackAngles(0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
int ProxyPixelTopology::channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir) const
{
   if (!this->surfaceDeformation()) return specificTopology().channel(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());
    
  return specificTopology().channel(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
std::pair<float,float> ProxyPixelTopology::pixel( const LocalPoint& lp ) const
{
  return this->pixel(lp, Topology::LocalTrackAngles(0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
std::pair<float,float> ProxyPixelTopology::pixel( const LocalPoint& lp,
						  const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().pixel(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());
  
  return specificTopology().pixel(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyPixelTopology::localX(const float mpX) const
{
  return this->localX(mpX, Topology::LocalTrackPred(0., 0., 0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
float ProxyPixelTopology::localX(const float mpX,
				 const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localX(mpX);
  
  // add correction from SurfaceDeformation
  float xOld = specificTopology().localX(mpX); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  
  return xOld + corr.x();
}

////////////////////////////////////////////////////////////////////////////////
float ProxyPixelTopology::localY(const float mpY) const
{
  return this->localY(mpY, Topology::LocalTrackPred(0., 0., 0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
float ProxyPixelTopology::localY(const float mpY,
				 const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localY(mpY);

  // add correction from SurfaceDeformation
  float yOld = specificTopology().localY(mpY); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  
  return yOld + corr.y();
}

////////////////////////////////////////////////////////////////////////////////
std::pair<float,float> ProxyPixelTopology::pitch() const { return specificTopology().pitch(); }

////////////////////////////////////////////////////////////////////////////////
int ProxyPixelTopology::nrows() const { return specificTopology().nrows(); }

////////////////////////////////////////////////////////////////////////////////
int ProxyPixelTopology::ncolumns() const { return specificTopology().ncolumns(); }

////////////////////////////////////////////////////////////////////////////////
const GeomDetType& ProxyPixelTopology::type() const { return *theType; }

////////////////////////////////////////////////////////////////////////////////
PixelGeomDetType& ProxyPixelTopology::specificType() const { return *theType; }

////////////////////////////////////////////////////////////////////////////////
const PixelTopology& ProxyPixelTopology::specificTopology() const
{ 
  return specificType().specificTopology();
}

////////////////////////////////////////////////////////////////////////////////
void ProxyPixelTopology::setSurfaceDeformation(const SurfaceDeformation * deformation)
{ 
  theSurfaceDeformation = deformation;
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyPixelTopology::positionCorrection(const LocalPoint &pos,
				       const Topology::LocalTrackAngles &dir) const
{
  const SurfaceDeformation::Local2DPoint pos2D(pos.x(), pos.y());// change precision and dimension

  return this->surfaceDeformation()->positionCorrection(pos2D, dir,
							theBounds.length(), theBounds.width());
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyPixelTopology::positionCorrection(const Topology::LocalTrackPred &trk) const
{
  return this->surfaceDeformation()->positionCorrection(trk.point(), trk.angles(),
							theBounds.length(), theBounds.width());
}

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "Geometry/TrackerGeometryBuilder/interface/ProxyStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

////////////////////////////////////////////////////////////////////////////////
ProxyStripTopology::ProxyStripTopology(StripGeomDetType* type, BoundPlane * bp)
  :theType(type), theBounds(bp->bounds())
{
  
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyStripTopology::localPosition( const MeasurementPoint& mp ) const
{
  return specificTopology().localPosition(mp);

// FIXME: Better this way? Well, but posOrig will not contain useful y!
//   if (!this->surfaceDeformation()) return specificTopology().localPosition(mp);
//
//   // correct with position information from input and zero track angles 
//   const LocalPoint posOrig(specificTopology().localPosition(mp));
//   return this->localPosition(mp, Topology::LocalTrackPred(posOrig.x(), posOrig.y(), 0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyStripTopology::localPosition( const MeasurementPoint& mp, 
					      const Topology::LocalTrackPred &trkPred ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPosition(mp);
  
  // add correction from SurfaceDeformation
  const LocalPoint posOld(specificTopology().localPosition(mp)); // 'original position'
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));

  return LocalPoint(posOld.x()+corr.x(), posOld.y()+corr.y(), posOld.z());
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyStripTopology::localPosition( float strip ) const
{
  return specificTopology().localPosition(strip);

// FIXME: Better this way? Well, but posOrig will not contain useful y!
//   if (!this->surfaceDeformation()) return specificTopology().localPosition(strip);

//   // correct with position information from input and zero track angles 
//   const LocalPoint posOrig(specificTopology().localPosition(strip));
//   return this->localPosition(mp, Topology::LocalTrackPred(posOrig.x(), posOrig.y(), 0., 0.));
}

////////////////////////////////////////////////////////////////////////////////
LocalPoint ProxyStripTopology::localPosition(float strip, const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPosition(strip);
  
  // add correction from SurfaceDeformation
  const LocalPoint posOld(specificTopology().localPosition(strip));

  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(trkPred));
  return LocalPoint(posOld.x()+corr.x(), posOld.y()+corr.y(), posOld.z());
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyStripTopology::localError( float strip, float stripErr2 ) const
{
  //  return this->localError(strip, stripErr2, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().localError(strip, stripErr2);

}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyStripTopology::localError(float strip, float stripErr2,
					  const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localError(strip, stripErr2);

  // FIXME: Add code to actually use SurfaceDeformation

  return specificTopology().localError(strip, stripErr2);
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyStripTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me) const
{
  //return this->localError(mp, me, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().localError(mp, me);
}

////////////////////////////////////////////////////////////////////////////////
LocalError ProxyStripTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me, 
					   const Topology::LocalTrackPred &trkPred) const
{
  if (!this->surfaceDeformation()) return specificTopology().localError(mp, me);

  // FIXME: Add code to actually use SurfaceDeformation

  return specificTopology().localError(mp, me);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementPoint ProxyStripTopology::measurementPosition( const LocalPoint& lp) const
{
  //  return this->measurementPosition(lp, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().measurementPosition(lp);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementPoint ProxyStripTopology::measurementPosition( const LocalPoint& lp, 
							  const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementPosition(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementPosition(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementError ProxyStripTopology::measurementError( const LocalPoint& lp, const LocalError& le ) const
{
  // return this->measurementError(lp, le, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().measurementError(lp, le);
}

////////////////////////////////////////////////////////////////////////////////
MeasurementError ProxyStripTopology::measurementError( const LocalPoint &lp, const LocalError &le,
						       const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementError(lp, le);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementError(posOrig, le);
}

////////////////////////////////////////////////////////////////////////////////
int ProxyStripTopology::channel( const LocalPoint& lp) const
{
  //  return this->channel(lp, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().channel(lp);
}

////////////////////////////////////////////////////////////////////////////////
int ProxyStripTopology::channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir) const
{
   if (!this->surfaceDeformation()) return specificTopology().channel(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());
    
  return specificTopology().channel(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::strip(const LocalPoint& lp) const
{
  //  return this->strip(lp, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().strip(lp);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::strip( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().strip(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().strip(posOrig);
}

float ProxyStripTopology::pitch() const
{
  return specificTopology().pitch();
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::localPitch( const LocalPoint& lp) const
{
  //  return this->localPitch(lp, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().localPitch(lp);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::localPitch( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPitch(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().localPitch(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::stripAngle( float strip ) const
{
  //  return this->stripAngle(strip, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().stripAngle(strip);
}

////////////////////////////////////////////////////////////////////////////////
int ProxyStripTopology::nstrips() const
{
  return specificTopology().nstrips();
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::stripLength() const
{
  return specificTopology().stripLength();
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::localStripLength(const LocalPoint& lp) const
{
  //return this->localStripLength(lp, Topology::LocalTrackAngles(0., 0.));
  return specificTopology().localStripLength(lp);
}

////////////////////////////////////////////////////////////////////////////////
float ProxyStripTopology::localStripLength( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localStripLength(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().localStripLength(posOrig);
}

////////////////////////////////////////////////////////////////////////////////
const GeomDetType& ProxyStripTopology::type() const { return *theType;}

////////////////////////////////////////////////////////////////////////////////
StripGeomDetType& ProxyStripTopology::specificType() const { return *theType;}

////////////////////////////////////////////////////////////////////////////////
const StripTopology& ProxyStripTopology::specificTopology() const
{ 
  return specificType().specificTopology();
}

////////////////////////////////////////////////////////////////////////////////
void ProxyStripTopology::setSurfaceDeformation(const SurfaceDeformation * deformation)
{ 
  theSurfaceDeformation = deformation;
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyStripTopology::positionCorrection(const LocalPoint &pos,
				       const Topology::LocalTrackAngles &dir) const
{
  const SurfaceDeformation::Local2DPoint pos2D(pos.x(), pos.y());// change precision and dimension

  return this->surfaceDeformation()->positionCorrection(pos2D, dir,
							theBounds.length(), theBounds.width());
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyStripTopology::positionCorrection(const Topology::LocalTrackPred &trk) const
{
  return this->surfaceDeformation()->positionCorrection(trk.point(), trk.angles(),
							theBounds.length(), theBounds.width());
}

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "Geometry/TrackerTopology/interface/ProxyStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

ProxyStripTopology::ProxyStripTopology(StripGeomDetType* type, BoundPlane * bp)
  :theType(type), theBounds(bp->bounds())
{
  
}

LocalPoint ProxyStripTopology::localPosition( const MeasurementPoint& mp ) const
{
  return this->localPosition(mp, Topology::LocalTrackAngles(0, 0));
}

LocalPoint ProxyStripTopology::localPosition( const MeasurementPoint& mp, 
					      const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPosition(mp);
 
  SurfaceDeformation::Local2DVector posCorr =
    this->surfaceDeformation()->positionCorrection(SurfaceDeformation::Local2DPoint(mp.x(), mp.y()),
						   dir,
						   theBounds.length(),
						   theBounds.width());
  LocalPoint localPos = specificTopology().localPosition(mp);

  // Do something with the correction
  
  return localPos;
}

LocalPoint ProxyStripTopology::localPosition( float strip ) const
{
  return this->localPosition(strip, Topology::LocalTrackAngles(0, 0));
}

LocalPoint ProxyStripTopology::localPosition( float strip, const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPosition(strip);
  
  // Add code to actually use SurfaceDeformation
  
  return specificTopology().localPosition(strip);
}

LocalError ProxyStripTopology::localError( float strip, float stripErr2 ) const
{
  return this->localError(strip, stripErr2, Topology::LocalTrackAngles(0, 0));
}

LocalError ProxyStripTopology::localError( float strip, float stripErr2, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localError(strip, stripErr2);

  // Add code to actually use SurfaceDeformation

  return specificTopology().localError(strip, stripErr2);
}

LocalError ProxyStripTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me) const
{
  return this->localError(mp, me, Topology::LocalTrackAngles(0, 0));
}

LocalError ProxyStripTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me, 
					   const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().localError(mp, me);

  // Add code to actually use SurfaceDeformation

  return specificTopology().localError(mp, me);
}

MeasurementPoint ProxyStripTopology::measurementPosition( const LocalPoint& lp) const
{
  return this->measurementPosition(lp, Topology::LocalTrackAngles(0, 0));
}

MeasurementPoint ProxyStripTopology::measurementPosition( const LocalPoint& lp, 
							  const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementPosition(lp);

  // Add code to actually use SurfaceDeformation

  return specificTopology().measurementPosition(lp);
}

MeasurementError ProxyStripTopology::measurementError( const LocalPoint& lp, const LocalError& le ) const
{
  return this->measurementError(lp, le, Topology::LocalTrackAngles(0, 0));
}

MeasurementError ProxyStripTopology::measurementError( const LocalPoint &lp, const LocalError &le,
						       const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementError(lp, le);

  // Add code to actually use SurfaceDeformation

  return specificTopology().measurementError(lp, le);
}

int ProxyStripTopology::channel( const LocalPoint& lp) const
{
  return this->channel(lp, Topology::LocalTrackAngles(0, 0));
}

int ProxyStripTopology::channel( const LocalPoint &lp, const Topology::LocalTrackAngles &dir) const
{
   if (!this->surfaceDeformation()) return specificTopology().channel(lp);

  // Add code to actually use SurfaceDeformation

   return specificTopology().channel(lp);
}

float ProxyStripTopology::strip(const LocalPoint& lp) const
{
  return this->strip(lp, Topology::LocalTrackAngles(0, 0));
}

float ProxyStripTopology::strip( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().strip(lp);

  // Add code to actually use SurfaceDeformation

  return specificTopology().strip(lp);
}

float ProxyStripTopology::pitch() const
{
  return specificTopology().pitch();
}

float ProxyStripTopology::localPitch( const LocalPoint& lp) const
{
  return this->localPitch(lp, Topology::LocalTrackAngles(0, 0));
}

float ProxyStripTopology::localPitch( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localPitch(lp);

  // Add code to actually use SurfaceDeformation

  return specificTopology().localPitch(lp);
}

float ProxyStripTopology::stripAngle( float strip ) const
{
  return this->stripAngle(strip, Topology::LocalTrackAngles(0, 0));
}

float ProxyStripTopology::stripAngle( float strip, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().stripAngle(strip);

  // Add code to actually use SurfaceDeformation

  return specificTopology().stripAngle(strip);
}

int ProxyStripTopology::nstrips() const
{
  return specificTopology().nstrips();
}

float ProxyStripTopology::stripLength() const
{
  return specificTopology().stripLength();
}

float ProxyStripTopology::localStripLength(const LocalPoint& lp) const
{
  return this->localStripLength(lp, Topology::LocalTrackAngles(0, 0));
}

float ProxyStripTopology::localStripLength( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localStripLength(lp);

  // Add code to actually use SurfaceDeformation

  return specificTopology().localStripLength(lp);
}

const GeomDetType& ProxyStripTopology::type() const { return *theType;}

StripGeomDetType& ProxyStripTopology::specificType() const { return *theType;}

const StripTopology& ProxyStripTopology::specificTopology() const { 
  return specificType().specificTopology();
}

void ProxyStripTopology::setSurfaceDeformation(const SurfaceDeformation * deformation)
{ 
  theSurfaceDeformation = deformation;
}

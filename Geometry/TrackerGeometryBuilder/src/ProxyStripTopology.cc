#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "Geometry/TrackerGeometryBuilder/interface/ProxyStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

////////////////////////////////////////////////////////////////////////////////
ProxyStripTopology::ProxyStripTopology(StripGeomDetType* type, BoundPlane * bp)
  :theType(type), theLength(bp->bounds().length()), theWidth(bp->bounds().width())
{
}

////////////////////////////////////////////////////////////////////////////////
/* inlined
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
*/

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
/* inlined
LocalPoint ProxyStripTopology::localPosition( float strip ) const
{
  return specificTopology().localPosition(strip);

// FIXME: Better this way? Well, but posOrig will not contain useful y!
//   if (!this->surfaceDeformation()) return specificTopology().localPosition(strip);

//   // correct with position information from input and zero track angles 
//   const LocalPoint posOrig(specificTopology().localPosition(strip));
//   return this->localPosition(mp, Topology::LocalTrackPred(posOrig.x(), posOrig.y(), 0., 0.));
}
*/

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
LocalError ProxyStripTopology::localError(float strip, float stripErr2,
					  const Topology::LocalTrackPred &trkPred) const
{
  // 'strip' is from measurement frame and the topology knows to
  // calculate the cartesian error.
  // But assuming no uncertainty on the SurfaceDeformation variables,
  // the errors do not change from a simple shift to compensate
  // that the track 'sees' the surface at another place than it thinks...
  
  // In case of TwoBowedSurfacesDeformation one could add corrections here due to 
  // relative rotations of the sensors...
  return specificTopology().localError(strip, stripErr2);
}


////////////////////////////////////////////////////////////////////////////////
LocalError ProxyStripTopology::localError( const MeasurementPoint& mp,
					   const MeasurementError& me, 
					   const Topology::LocalTrackPred &trkPred) const
{
  // See comment in localError(float strip, float stripErr2,
  //                           const Topology::LocalTrackPred &trkPred)!
  return specificTopology().localError(mp, me);
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
MeasurementError ProxyStripTopology::measurementError( const LocalPoint &lp, const LocalError &le,
						       const Topology::LocalTrackAngles &dir) const
{
  if (!this->surfaceDeformation()) return specificTopology().measurementError(lp, le);

  // assuming 'lp' comes from a track prediction
  // (i.e. where the track thinks it hits the surface)
  // we need to subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().measurementError(posOrig, le);
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
float ProxyStripTopology::strip( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().strip(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().strip(posOrig);
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
float ProxyStripTopology::localStripLength( const LocalPoint& lp, const Topology::LocalTrackAngles &dir ) const
{
  if (!this->surfaceDeformation()) return specificTopology().localStripLength(lp);

  // subtract correction from SurfaceDeformation
  const SurfaceDeformation::Local2DVector corr(this->positionCorrection(lp, dir));
  const LocalPoint posOrig(lp.x() - corr.x(), lp.y() - corr.y(), lp.z());

  return specificTopology().localStripLength(posOrig);
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
							theLength, theWidth);
}

////////////////////////////////////////////////////////////////////////////////
SurfaceDeformation::Local2DVector
ProxyStripTopology::positionCorrection(const Topology::LocalTrackPred &trk) const
{
  return this->surfaceDeformation()->positionCorrection(trk.point(), trk.angles(),
							theLength, theWidth);
}

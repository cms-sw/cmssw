#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

MagVolume6Faces::MagVolume6Faces( const PositionType& pos,
				  const RotationType& rot, 
				  DDSolidShape shape,
				  const std::vector<VolumeSide>& faces,
				  const MagneticFieldProvider<float> * mfp)
  : MagVolume(pos,rot,shape,mfp), theFaces(faces), copyno(0) 
{}

bool MagVolume6Faces::inside( const GlobalPoint& gp, double tolerance) const 
{

  // check if the point is on the correct side of all delimiting surfaces
  for (std::vector<VolumeSide>::const_iterator i=theFaces.begin(); i!=theFaces.end(); ++i) {
    Surface::Side side = i->surface().side( gp, tolerance);
    if ( side != i->surfaceSide() && side != SurfaceOrientation::onSurface) return false;
  }
  return true;
}

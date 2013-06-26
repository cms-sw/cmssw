// 
//#include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/VolumeGeometry/interface/MagCylinder.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"

MagCylinder::MagCylinder( const PositionType& pos,
			  const RotationType& rot, 
			  DDSolidShape shape,
			  const std::vector<VolumeSide>& faces,
			  const MagneticFieldProvider<float> * mfp)
  : MagVolume(pos,rot,shape,mfp), theFaces(faces), theZmin(0.), theZmax(0.), theInnerR(0.), theOuterR(0.)
{
  using SurfaceOrientation::GlobalFace;

  unsigned int def = 0;
  for (std::vector<VolumeSide>::const_iterator i=faces.begin(); i!= faces.end(); ++i) {
    if (i->globalFace() == SurfaceOrientation::zminus) {
      theZmin = MagVolume::toLocal( i->surface().position()).z();
      ++def;
    }
    else if (i->globalFace() == SurfaceOrientation::zplus) {
      theZmax = MagVolume::toLocal( i->surface().position()).z();
      ++def;
    }
    else if (i->globalFace() == SurfaceOrientation::outer || i->globalFace() == SurfaceOrientation::inner) {
      const Cylinder* cyl = dynamic_cast<const Cylinder*>(&(i->surface()));
      if (cyl == 0) {
	throw MagGeometryError("MagCylinder inner/outer surface is not a cylinder");
      }
      if (i->globalFace() == SurfaceOrientation::outer) theOuterR = cyl->radius();
      else                                              theInnerR = cyl->radius();
      ++def;
    }
  }
  if (def != faces.size()) {
    throw MagGeometryError("MagCylinder constructed with wrong number/type of faces");
  }
  
}

bool MagCylinder::inside( const GlobalPoint& gp, double tolerance) const 
{
  return inside( toLocal(gp), tolerance);
}

bool MagCylinder::inside( const LocalPoint& lp, double tolerance) const 
{
  Scalar r( lp.perp());
  return 
    lp.z() > theZmin - tolerance &&
    lp.z() < theZmax + tolerance &&
    r      > theInnerR - tolerance && 
    r      < theOuterR + tolerance;
}

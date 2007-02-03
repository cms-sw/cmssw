// 
//#include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/VolumeGeometry/interface/MagCylinder.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"

#ifndef NO_DETAILED_TIMING
// TOFIX
// #include "Utilities/Notification/interface/TimingReport.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"
#endif

MagCylinder::MagCylinder( const PositionType& pos,
			  const RotationType& rot, 
			  DDSolidShape shape,
			  const std::vector<VolumeSide>& faces,
			  const MagneticFieldProvider<float> * mfp)
  : MagVolume(pos,rot,shape,mfp), theFaces(faces), theInnerR(0)
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
  
#ifndef NO_DETAILED_TIMING
// TOFIX
//   static SimpleConfigurable<bool> timerOn(false,"MagCylinder:timing");
//   bool timerOn = false;
//   (*TimingReport::current()).switchOn("MagCylinder::inside",timerOn);
#endif
}

bool MagCylinder::inside( const GlobalPoint& gp, double tolerance) const 
{
#ifndef NO_DETAILED_TIMING
// TOFIX
//   static TimingReport::Item & timer = (*TimingReport::current())["MagCylinder::inside(global)"];
//   TimeMe t(timer,false);
#endif
  return inside( toLocal(gp), tolerance);
}

bool MagCylinder::inside( const LocalPoint& lp, double tolerance) const 
{
#ifndef NO_DETAILED_TIMING
// TOFIX
//   static TimingReport::Item & timer = (*TimingReport::current())["MagCylinder::inside(local)"];
//   TimeMe t(timer,false);
#endif

  Scalar r( lp.perp());
  return 
    lp.z() > theZmin - tolerance &&
    lp.z() < theZmax + tolerance &&
    r      > theInnerR - tolerance && 
    r      < theOuterR + tolerance;
}

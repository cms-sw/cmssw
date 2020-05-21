#include "MagneticField/VolumeGeometry/interface/MagCylinder.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"

MagCylinder::MagCylinder(const PositionType& pos,
                         const RotationType& rot,
                         const std::vector<VolumeSide>& faces,
                         const MagneticFieldProvider<float>* mfp)
    : MagVolume(pos, rot, mfp), theFaces(faces), theZmin(0.), theZmax(0.), theInnerR(0.), theOuterR(0.) {
  using SurfaceOrientation::GlobalFace;

  unsigned int def = 0;
  for (const auto& face : faces) {
    if (face.globalFace() == SurfaceOrientation::zminus) {
      theZmin = MagVolume::toLocal(face.surface().position()).z();
      ++def;
    } else if (face.globalFace() == SurfaceOrientation::zplus) {
      theZmax = MagVolume::toLocal(face.surface().position()).z();
      ++def;
    } else if (face.globalFace() == SurfaceOrientation::outer || face.globalFace() == SurfaceOrientation::inner) {
      const Cylinder* cyl = dynamic_cast<const Cylinder*>(&(face.surface()));
      if (cyl == nullptr) {
        throw MagGeometryError("MagCylinder inner/outer surface is not a cylinder");
      }
      if (face.globalFace() == SurfaceOrientation::outer)
        theOuterR = cyl->radius();
      else
        theInnerR = cyl->radius();
      ++def;
    }
  }
  if (def != faces.size()) {
    throw MagGeometryError("MagCylinder constructed with wrong number/type of faces");
  }
}

bool MagCylinder::inside(const GlobalPoint& gp, double tolerance) const { return inside(toLocal(gp), tolerance); }

bool MagCylinder::inside(const LocalPoint& lp, double tolerance) const {
  Scalar r(lp.perp());
  return lp.z() > theZmin - tolerance && lp.z() < theZmax + tolerance && r > theInnerR - tolerance &&
         r < theOuterR + tolerance;
}

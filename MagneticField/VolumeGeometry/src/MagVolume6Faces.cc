#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

MagVolume6Faces::MagVolume6Faces(const PositionType& pos,
                                 const RotationType& rot,
                                 const std::vector<VolumeSide>& faces,
                                 const MagneticFieldProvider<float>* mfp,
                                 double sf)
    : MagVolume(pos, rot, mfp, sf), volumeNo(0), copyno(0), theFaces(faces) {}

bool MagVolume6Faces::inside(const GlobalPoint& gp, double tolerance) const {
  // check if the point is on the correct side of all delimiting surfaces
  for (const auto& theFace : theFaces) {
    Surface::Side side = theFace.surface().side(gp, tolerance);
    if (side != theFace.surfaceSide() && side != SurfaceOrientation::onSurface)
      return false;
  }
  return true;
}

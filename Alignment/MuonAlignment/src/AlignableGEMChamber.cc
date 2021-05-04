/* AlignableGEMChamber
 * \author Hyunyong Kim - TAMU
 */
#include "Alignment/MuonAlignment/interface/AlignableGEMChamber.h"

AlignableGEMChamber::AlignableGEMChamber(const GeomDet* geomDet) : AlignableDet(geomDet) {
  theStructureType = align::AlignableGEMChamber;
  theSurface = geomDet->surface();
}

void AlignableGEMChamber::update(const GeomDet* geomDet) {
  AlignableDet::update(geomDet);
  theSurface = geomDet->surface();
}

std::ostream& operator<<(std::ostream& os, const AlignableGEMChamber& r) {
  const auto& theDets = r.components();

  os << "    This GEMChamber contains " << theDets.size() << " units" << std::endl;
  os << "    position = " << r.globalPosition() << std::endl;
  os << "    (phi, r, z)= (" << r.globalPosition().phi() << "," << r.globalPosition().perp() << ","
     << r.globalPosition().z();
  os << "), orientation:" << std::endl << r.globalRotation() << std::endl;

  os << "    total displacement and rotation: " << r.displacement() << std::endl;
  os << r.rotation() << std::endl;

  for (const auto& idet : theDets) {
    const auto& comp = idet->components();

    for (unsigned int i = 0; i < comp.size(); ++i) {
      os << "     Det position, phi, r: " << comp[i]->globalPosition() << " , " << comp[i]->globalPosition().phi()
         << " , " << comp[i]->globalPosition().perp() << std::endl;
      os << "     local  position, phi, r: " << r.surface().toLocal(comp[i]->globalPosition()) << " , "
         << r.surface().toLocal(comp[i]->globalPosition()).phi() << " , "
         << r.surface().toLocal(comp[i]->globalPosition()).perp() << std::endl;
    }
  }

  return os;
}

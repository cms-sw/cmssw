/** \file
 *
 *  $Date: 2008/03/26 21:59:25 $
 *  $Revision: 1.10 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableDTSuperLayer.h"

AlignableDTChamber::AlignableDTChamber(const GeomDet* geomDet) : AlignableDet(geomDet, false) {
  // even though we overload alignableObjectId(), it's dangerous to
  // have two different claims about the structure type
  theStructureType = align::AlignableDTChamber;

  // The unique thing about DT chambers is that they are Dets that contain Dets (superlayers)
  // The superlayer Dets contain DetUnits (layers), as usual
  const std::vector<const GeomDet*>& geomDets = geomDet->components();
  for (std::vector<const GeomDet*>::const_iterator idet = geomDets.begin(); idet != geomDets.end(); ++idet) {
    addComponent(new AlignableDTSuperLayer(*idet));
  }

  // DO NOT let the chamber position become an average of the layers
  this->theSurface = geomDet->surface();
}

/// Printout the DetUnits in the DT chamber
std::ostream& operator<<(std::ostream& os, const AlignableDTChamber& r) {
  const auto& theDets = r.components();

  os << "    This DTChamber contains " << theDets.size() << " units" << std::endl;
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

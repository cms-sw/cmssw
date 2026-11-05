/** \file
 *
 *  $Date: 2024/12/26 21:59:25 $
 *  $Revision: 1.0 $
 *  \author Pablo Martínez Ruiz de Arbol - IFCA
 */

#include "Alignment/MTDAlignment/interface/AlignableBTLSensorModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlignableBTLSensorModule::AlignableBTLSensorModule(const GeomDet* geomDet) : AlignableDet(geomDet, false) {
  // even though we overload alignableObjectId(), it's dangerous to
  // have two different claims about the structure type
  theStructureType = align::AlignableBTLSensorModule;

  //////// Probably we don't need this in the BTL, the crystals are unitDets
  //const std::vector<const GeomDet*>& geomDets = geomDet->components();
  //for (std::vector<const GeomDet*>::const_iterator idet = geomDets.begin(); idet != geomDets.end(); ++idet) {
  //  addComponent(new AlignableBTLSensorModule(*idet));
  //}

  this->theSurface = geomDet->surface();
}

/// Printout the DetUnits in the BTL Module
std::ostream& operator<<(std::ostream& os, const AlignableBTLSensorModule& r) {
  const auto& theDets = r.components();

  os << "    This BTL Sensor Module contains " << theDets.size() << " units" << std::endl;
  os << "    position = " << r.globalPosition() << std::endl;
  os << "    (phi, r, z)= (" << r.globalPosition().phi() << "," << r.globalPosition().perp() << ","
     << r.globalPosition().z();
  os << "), orientation:" << std::endl << r.globalRotation() << std::endl;

  os << "    total displacement and rotation: " << r.displacement() << std::endl;
  os << r.rotation() << std::endl;

  return os;
}

void AlignableBTLSensorModule::dump(void) const { edm::LogInfo("AlignableDump") << (*this); }

/** \file
 *
 *  $Date: 2024/12/26 21:59:25 $
 *  $Revision: 1.0 $
 *  \author Pablo Martínez Ruiz de Arbol - IFCA
 */

#include "Alignment/MTDAlignment/interface/AlignableETLSensor.h"

AlignableETLSensor::AlignableETLSensor(const GeomDet* geomDet) : AlignableDet(geomDet, false) {
  // even though we overload alignableObjectId(), it's dangerous to
  // have two different claims about the structure type
  theStructureType = align::AlignableETLSensor;

  // DO NOT let the position become an average of the layers
  this->theSurface = geomDet->surface();
}

/// Printout the DetUnits in the ETL Sensor
std::ostream& operator<<(std::ostream& os, const AlignableETLSensor& r) {
  const auto& theDets = r.components();

  os << "    This ETLSensor contains " << theDets.size() << " units" << std::endl;
  os << "    position = " << r.globalPosition() << std::endl;
  os << "    (phi, r, z)= (" << r.globalPosition().phi() << "," << r.globalPosition().perp() << ","
     << r.globalPosition().z();
  os << "), orientation:" << std::endl << r.globalRotation() << std::endl;

  os << "    total displacement and rotation: " << r.displacement() << std::endl;
  os << r.rotation() << std::endl;

  return os;
}

/// Recursive printout of whole structure
void AlignableETLSensor::dump(void) const { edm::LogInfo("AlignableDump") << (*this); }

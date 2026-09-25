/** \file
 *
 *  $Date: 2024/12/10 16:36:41 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/MTDAlignment/interface/AlignableBTLModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of BTL RUs and computes the surface from them
AlignableBTLModule::AlignableBTLModule(const std::vector<AlignableBTLSensorModule*>& btlSensorModules)
    : AlignableComposite(btlSensorModules[0]->id(), align::AlignableBTLModule) {
  theBTLSensorModules.insert(theBTLSensorModules.end(), btlSensorModules.begin(), btlSensorModules.end());

  // maintain also list of components
  for (const auto& bmodule : btlSensorModules) {
    const auto mother = bmodule->mother();
    this->addComponent(bmodule);  // components will be deleted by dtor of AlignableComposite
    bmodule->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return Alignable module at given index
AlignableBTLSensorModule& AlignableBTLModule::sensormod(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Module index (" << i << ") out of range";

  return *theBTLSensorModules[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableBTLModule::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

/// Compute average z position from all components (x and y forced to 0)
AlignableBTLModule::PositionType AlignableBTLModule::computePosition() {
  float xx = 0.;
  float yy = 0.;
  float zz = 0.;

  for (std::vector<AlignableBTLSensorModule*>::iterator imodule = theBTLSensorModules.begin();
       imodule != theBTLSensorModules.end();
       imodule++) {
    xx += (*imodule)->globalPosition().x();
    yy += (*imodule)->globalPosition().y();
    zz += (*imodule)->globalPosition().z();
  }
  xx /= static_cast<float>(theBTLSensorModules.size());
  yy /= static_cast<float>(theBTLSensorModules.size());
  zz /= static_cast<float>(theBTLSensorModules.size());

  return PositionType(xx, yy, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableBTLModule::RotationType AlignableBTLModule::computeOrientation() { return RotationType(); }

/// Output Station information
std::ostream& operator<<(std::ostream& os, const AlignableBTLModule& b) {
  os << "This BTL module contains " << b.theBTLSensorModules.size() << " BTL sensor modules" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole DT Station structure
void AlignableBTLModule::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableBTLSensorModule*>::const_iterator iModule = theBTLSensorModules.begin();
       iModule != theBTLSensorModules.end();
       iModule++)
    edm::LogInfo("AlignableDump") << (**iModule);
}

/** \file
 *
 *  $Date: 2024/12/10 16:36:41 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/MTDAlignment/interface/AlignableBTLRU.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of BTL RUs and computes the surface from them
AlignableBTLRU::AlignableBTLRU(const std::vector<AlignableBTLModule*>& btlModules)
    : AlignableComposite(btlModules[0]->id(), align::AlignableBTLRU) {
  theBTLModules.insert(theBTLModules.end(), btlModules.begin(), btlModules.end());

  // maintain also list of components
  for (const auto& bmodule : btlModules) {
    const auto mother = bmodule->mother();
    this->addComponent(bmodule);  // components will be deleted by dtor of AlignableComposite
    bmodule->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return Alignable module at given index
AlignableBTLModule& AlignableBTLRU::mod(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Module index (" << i << ") out of range";

  return *theBTLModules[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableBTLRU::computeSurface() { return AlignableSurface(computePosition(), computeOrientation()); }

/// Compute average z position from all components (x and y forced to 0)
AlignableBTLRU::PositionType AlignableBTLRU::computePosition() {
  float xx = 0.;
  float yy = 0.;
  float zz = 0.;

  for (std::vector<AlignableBTLModule*>::iterator imodule = theBTLModules.begin(); imodule != theBTLModules.end();
       imodule++) {
    xx += (*imodule)->globalPosition().x();
    yy += (*imodule)->globalPosition().y();
    zz += (*imodule)->globalPosition().z();
  }
  xx /= static_cast<float>(theBTLModules.size());
  yy /= static_cast<float>(theBTLModules.size());
  zz /= static_cast<float>(theBTLModules.size());

  return PositionType(xx, yy, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableBTLRU::RotationType AlignableBTLRU::computeOrientation() { return RotationType(); }

/// Output Station information
std::ostream& operator<<(std::ostream& os, const AlignableBTLRU& b) {
  os << "This BTL RU contains " << b.theBTLModules.size() << " BTL modules" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole DT Station structure
void AlignableBTLRU::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableBTLModule*>::const_iterator iModule = theBTLModules.begin(); iModule != theBTLModules.end();
       iModule++)
    edm::LogInfo("AlignableDump") << (**iModule);
}

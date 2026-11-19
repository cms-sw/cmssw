/** \file
 *
 *  $Date: 27 Oct 2024  $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/MTDAlignment/interface/AlignableETLDisk.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of stations and computes the surface from them
AlignableETLDisk::AlignableETLDisk(const std::vector<AlignableETLDee*>& etlDees)
    : AlignableComposite(etlDees[0]->id(), align::AlignableETLDisk) {
  theETLDees.insert(theETLDees.end(), etlDees.begin(), etlDees.end());

  // maintain also list of components
  for (const auto& dee : etlDees) {
    const auto mother = dee->mother();
    this->addComponent(dee);  // components will be deleted by dtor of AlignableComposite
    dee->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return Alignable disk at given index
AlignableETLDee& AlignableETLDisk::dee(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Station index (" << i << ") out of range";

  return *theETLDees[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableETLDisk::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

/// Compute average z position from all components (x and y forced to 0)
AlignableETLDisk::PositionType AlignableETLDisk::computePosition() {
  float zz = 0.;
  float xx = 0.;
  float yy = 0.;
  for (std::vector<AlignableETLDee*>::iterator ilayer = theETLDees.begin(); ilayer != theETLDees.end(); ilayer++) {
    xx += (*ilayer)->globalPosition().x();
    yy += (*ilayer)->globalPosition().y();
    zz += (*ilayer)->globalPosition().z();
  }
  xx /= static_cast<float>(theETLDees.size());
  yy /= static_cast<float>(theETLDees.size());
  zz /= static_cast<float>(theETLDees.size());

  return PositionType(xx, yy, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableETLDisk::RotationType AlignableETLDisk::computeOrientation() { return RotationType(); }

/// Output Wheel information
std::ostream& operator<<(std::ostream& os, const AlignableETLDisk& b) {
  os << "This ETLDisk contains " << b.theETLDees.size() << " ETL Dees" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole Tray structure
void AlignableETLDisk::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableETLDee*>::const_iterator iDee = theETLDees.begin(); iDee != theETLDees.end(); iDee++)
    (*iDee)->dump();
}

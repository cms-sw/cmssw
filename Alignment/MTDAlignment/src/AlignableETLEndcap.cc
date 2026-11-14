/** \file
 *
 *  $Date: 27 Oct 2024  $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/MTDAlignment/interface/AlignableETLEndcap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of stations and computes the surface from them
AlignableETLEndcap::AlignableETLEndcap(const std::vector<AlignableETLDisk*>& etlDisks)
    : AlignableComposite(etlDisks[0]->id(), align::AlignableETLEndcap) {
  theETLDisks.insert(theETLDisks.end(), etlDisks.begin(), etlDisks.end());

  // maintain also list of components
  for (const auto& disk : etlDisks) {
    const auto mother = disk->mother();
    this->addComponent(disk);  // components will be deleted by dtor of AlignableComposite
    disk->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return Alignable RU at given index
AlignableETLDisk& AlignableETLEndcap::disk(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Station index (" << i << ") out of range";

  return *theETLDisks[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableETLEndcap::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

/// Compute average z position from all components (x and y forced to 0)
AlignableETLEndcap::PositionType AlignableETLEndcap::computePosition() {
  float zz = 0.;
  for (std::vector<AlignableETLDisk*>::iterator ilayer = theETLDisks.begin(); ilayer != theETLDisks.end(); ilayer++) {
    zz += (*ilayer)->globalPosition().z();
  }
  zz /= static_cast<float>(theETLDisks.size());

  return PositionType(0.0, 0.0, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableETLEndcap::RotationType AlignableETLEndcap::computeOrientation() { return RotationType(); }

/// Output Wheel information
std::ostream& operator<<(std::ostream& os, const AlignableETLEndcap& b) {
  os << "This ETL Endcap " << b.theETLDisks.size() << " ETL Disks" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole Tray structure
void AlignableETLEndcap::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableETLDisk*>::const_iterator iDisk = theETLDisks.begin(); iDisk != theETLDisks.end(); iDisk++)
    (*iDisk)->dump();
}

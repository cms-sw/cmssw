/** \file
 *
 *  $Date: 2008/04/10 16:36:41 $
 *  $Revision: 1.5 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of DT Chambers and computes the surface from them
AlignableDTStation::AlignableDTStation(const std::vector<AlignableDTChamber*>& dtChambers)
    : AlignableComposite(dtChambers[0]->id(), align::AlignableDTStation) {
  theDTChambers.insert(theDTChambers.end(), dtChambers.begin(), dtChambers.end());

  // maintain also list of components
  for (const auto& chamber : dtChambers) {
    const auto mother = chamber->mother();
    this->addComponent(chamber);  // components will be deleted by dtor of AlignableComposite
    chamber->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return Alignable DT Chamber at given index
AlignableDTChamber& AlignableDTStation::chamber(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "DT Chamber index (" << i << ") out of range";

  return *theDTChambers[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableDTStation::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

/// Compute average z position from all components (x and y forced to 0)
AlignableDTStation::PositionType AlignableDTStation::computePosition() {
  float zz = 0.;

  for (std::vector<AlignableDTChamber*>::iterator ilayer = theDTChambers.begin(); ilayer != theDTChambers.end();
       ilayer++)
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theDTChambers.size());

  return PositionType(0.0, 0.0, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableDTStation::RotationType AlignableDTStation::computeOrientation() { return RotationType(); }

/// Output Station information
std::ostream& operator<<(std::ostream& os, const AlignableDTStation& b) {
  os << "This DT Station contains " << b.theDTChambers.size() << " DT chambers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole DT Station structure
void AlignableDTStation::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableDTChamber*>::const_iterator iChamber = theDTChambers.begin();
       iChamber != theDTChambers.end();
       iChamber++)
    edm::LogInfo("AlignableDump") << (**iChamber);
}

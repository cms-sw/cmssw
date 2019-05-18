/** \file
 *
 *  $Date: 2008/04/10 16:36:41 $
 *  $Revision: 1.5 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of stations and computes the surface from them
AlignableDTWheel::AlignableDTWheel(const std::vector<AlignableDTStation*>& dtStations)
    : AlignableComposite(dtStations[0]->id(), align::AlignableDTWheel) {
  theDTStations.insert(theDTStations.end(), dtStations.begin(), dtStations.end());

  // maintain also list of components
  for (const auto& station : dtStations) {
    const auto mother = station->mother();
    this->addComponent(station);  // components will be deleted by dtor of AlignableComposite
    station->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return Alignable DT Station at given index
AlignableDTStation& AlignableDTWheel::station(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Station index (" << i << ") out of range";

  return *theDTStations[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableDTWheel::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

/// Compute average z position from all components (x and y forced to 0)
AlignableDTWheel::PositionType AlignableDTWheel::computePosition() {
  float zz = 0.;

  for (std::vector<AlignableDTStation*>::iterator ilayer = theDTStations.begin(); ilayer != theDTStations.end();
       ilayer++)
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theDTStations.size());

  return PositionType(0.0, 0.0, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableDTWheel::RotationType AlignableDTWheel::computeOrientation() { return RotationType(); }

/// Output Wheel information
std::ostream& operator<<(std::ostream& os, const AlignableDTWheel& b) {
  os << "This DTWheel contains " << b.theDTStations.size() << " DT stations" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole DT Wheel structure
void AlignableDTWheel::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableDTStation*>::const_iterator iStation = theDTStations.begin();
       iStation != theDTStations.end();
       iStation++)
    (*iStation)->dump();
}

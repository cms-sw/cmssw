/* AlignableGEMStation
 * \author Hyunyong Kim - TAMU
 */
#include "Alignment/MuonAlignment/interface/AlignableGEMStation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlignableGEMStation::AlignableGEMStation(const std::vector<AlignableGEMRing*>& GEMRings)
    : AlignableComposite(GEMRings[0]->id(), align::AlignableGEMStation) {
  theGEMRings.insert(theGEMRings.end(), GEMRings.begin(), GEMRings.end());

  for (const auto& ring : GEMRings) {
    const auto mother = ring->mother();
    this->addComponent(ring);
    ring->setMother(mother);
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

AlignableGEMRing& AlignableGEMStation::ring(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "GEM Ring index (" << i << ") out of range";

  return *theGEMRings[i];
}

AlignableSurface AlignableGEMStation::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

AlignableGEMStation::PositionType AlignableGEMStation::computePosition() {
  float zz = 0.;

  for (std::vector<AlignableGEMRing*>::iterator ilayer = theGEMRings.begin(); ilayer != theGEMRings.end(); ilayer++)
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theGEMRings.size());

  return PositionType(0.0, 0.0, zz);
}

AlignableGEMStation::RotationType AlignableGEMStation::computeOrientation() { return RotationType(); }

std::ostream& operator<<(std::ostream& os, const AlignableGEMStation& b) {
  os << "This GEM Station contains " << b.theGEMRings.size() << " GEM rings" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

void AlignableGEMStation::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableGEMRing*>::const_iterator iRing = theGEMRings.begin(); iRing != theGEMRings.end(); iRing++)
    edm::LogInfo("AlignableDump") << (**iRing);
}

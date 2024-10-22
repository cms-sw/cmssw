/* AlignableGEMRing
 * \author Hyunyong Kim - TAMU
 */
#include "Alignment/MuonAlignment/interface/AlignableGEMRing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlignableGEMRing::AlignableGEMRing(const std::vector<AlignableGEMSuperChamber*>& GEMSuperChambers)
    : AlignableComposite(GEMSuperChambers[0]->id(), align::AlignableGEMRing) {
  theGEMSuperChambers.insert(theGEMSuperChambers.end(), GEMSuperChambers.begin(), GEMSuperChambers.end());

  for (const auto& chamber : GEMSuperChambers) {
    const auto mother = chamber->mother();
    this->addComponent(chamber);
    chamber->setMother(mother);
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

AlignableGEMSuperChamber& AlignableGEMRing::superChamber(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "GEM Super Chamber index (" << i << ") out of range";

  return *theGEMSuperChambers[i];
}

AlignableSurface AlignableGEMRing::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

AlignableGEMRing::PositionType AlignableGEMRing::computePosition() {
  float zz = 0.;

  for (std::vector<AlignableGEMSuperChamber*>::iterator ichamber = theGEMSuperChambers.begin();
       ichamber != theGEMSuperChambers.end();
       ichamber++)
    zz += (*ichamber)->globalPosition().z();

  zz /= static_cast<float>(theGEMSuperChambers.size());

  return PositionType(0.0, 0.0, zz);
}

AlignableGEMRing::RotationType AlignableGEMRing::computeOrientation() { return RotationType(); }

std::ostream& operator<<(std::ostream& os, const AlignableGEMRing& b) {
  os << "This GEM Ring contains " << b.theGEMSuperChambers.size() << " GEM Super chambers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

void AlignableGEMRing::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableGEMSuperChamber*>::const_iterator iChamber = theGEMSuperChambers.begin();
       iChamber != theGEMSuperChambers.end();
       iChamber++)
    edm::LogInfo("AlignableDump") << (**iChamber);
}

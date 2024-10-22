/* AlignableGEMEndcap
 * \author Hyunyong Kim - TAMU
 */
#include <memory>

#include "Alignment/MuonAlignment/interface/AlignableGEMEndcap.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlignableGEMEndcap::AlignableGEMEndcap(const std::vector<AlignableGEMStation*>& GEMStations)
    : AlignableComposite(GEMStations[0]->id(), align::AlignableGEMEndcap) {
  theGEMStations.insert(theGEMStations.end(), GEMStations.begin(), GEMStations.end());

  for (const auto& station : GEMStations) {
    const auto mother = station->mother();
    this->addComponent(station);
    station->setMother(mother);
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

AlignableGEMStation& AlignableGEMEndcap::station(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Station index (" << i << ") out of range";

  return *theGEMStations[i];
}

AlignableSurface AlignableGEMEndcap::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

AlignableGEMEndcap::PositionType AlignableGEMEndcap::computePosition() {
  float zz = 0.;

  for (std::vector<AlignableGEMStation*>::iterator ilayer = theGEMStations.begin(); ilayer != theGEMStations.end();
       ilayer++)
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theGEMStations.size());

  return PositionType(0.0, 0.0, zz);
}

AlignableGEMEndcap::RotationType AlignableGEMEndcap::computeOrientation() { return RotationType(); }

std::ostream& operator<<(std::ostream& os, const AlignableGEMEndcap& b) {
  os << "This EndCap contains " << b.theGEMStations.size() << " GEM stations" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

void AlignableGEMEndcap::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableGEMStation*>::const_iterator iLayer = theGEMStations.begin();
       iLayer != theGEMStations.end();
       iLayer++)
    (*iLayer)->dump();
}

Alignments* AlignableGEMEndcap::alignments(void) const {
  Alignments* m_alignments = new Alignments();

  for (const auto& i : this->components()) {
    std::unique_ptr<Alignments> tmpAlignments{i->alignments()};
    std::copy(tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), std::back_inserter(m_alignments->m_align));
  }

  std::sort(m_alignments->m_align.begin(), m_alignments->m_align.end());

  return m_alignments;
}

AlignmentErrorsExtended* AlignableGEMEndcap::alignmentErrors(void) const {
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  for (const auto& i : this->components()) {
    std::unique_ptr<AlignmentErrorsExtended> tmpAlignmentErrorsExtended{i->alignmentErrors()};
    std::copy(tmpAlignmentErrorsExtended->m_alignError.begin(),
              tmpAlignmentErrorsExtended->m_alignError.end(),
              std::back_inserter(m_alignmentErrors->m_alignError));
  }

  std::sort(m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end());

  return m_alignmentErrors;
}

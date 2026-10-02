/** \file
 *
 *  $Date: 2024/12/15 16:36:41 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include <memory>

#include "Alignment/MTDAlignment/interface/AlignableBTL.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of trays and computes the surface from them
AlignableBTL::AlignableBTL(const std::vector<AlignableBTLTray*>& btlTrays)
    : AlignableComposite(btlTrays[0]->id(), align::AlignableBTL) {
  theBTLTrays.insert(theBTLTrays.end(), btlTrays.begin(), btlTrays.end());

  // maintain also list of components
  for (const auto& tray : btlTrays) {
    const auto mother = tray->mother();
    this->addComponent(tray);  // components will be deleted by dtor of AlignableComposite
    tray->setMother(mother);   // restore previous behaviour where mother is not set
  }

  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return AlignableBarrelLayer at given index
AlignableBTLTray& AlignableBTL::tray(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Tray index (" << i << ") out of range";

  return *theBTLTrays[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableBTL::computeSurface() { return AlignableSurface(computePosition(), computeOrientation()); }

/// Compute average z position from all components (x and y forced to 0)
AlignableBTL::PositionType AlignableBTL::computePosition() {
  float zz = 0.;

  for (std::vector<AlignableBTLTray*>::iterator itray = theBTLTrays.begin(); itray != theBTLTrays.end(); itray++)
    zz += (*itray)->globalPosition().z();

  zz /= static_cast<float>(theBTLTrays.size());

  return PositionType(0.0, 0.0, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableBTL::RotationType AlignableBTL::computeOrientation() { return RotationType(); }

/// Output Half Barrel information
std::ostream& operator<<(std::ostream& os, const AlignableBTL& b) {
  os << "This BTL contains " << b.theBTLTrays.size() << " BTL Trays" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole Half Barrel structure
void AlignableBTL::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableBTLTray*>::const_iterator iTray = theBTLTrays.begin(); iTray != theBTLTrays.end(); iTray++)
    (*iTray)->dump();
}

//__________________________________________________________________________________________________
Alignments* AlignableBTL::alignments(void) const {
  Alignments* m_alignments = new Alignments();

  // Add components recursively
  for (const auto& i : this->components()) {
    std::unique_ptr<Alignments> tmpAlignments{i->alignments()};
    std::copy(tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), std::back_inserter(m_alignments->m_align));
  }

  // sort by rawId
  std::sort(m_alignments->m_align.begin(), m_alignments->m_align.end());

  return m_alignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableBTL::alignmentErrors(void) const {
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for (const auto& i : this->components()) {
    std::unique_ptr<AlignmentErrorsExtended> tmpAlignmentErrorsExtended{i->alignmentErrors()};
    std::copy(tmpAlignmentErrorsExtended->m_alignError.begin(),
              tmpAlignmentErrorsExtended->m_alignError.end(),
              std::back_inserter(m_alignmentErrors->m_alignError));
  }

  // sort by rawId
  std::sort(m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end());

  return m_alignmentErrors;
}

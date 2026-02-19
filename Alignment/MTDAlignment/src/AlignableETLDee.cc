/** \file
 *
 *  $Date: 2024/12/10 16:36:41 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include <memory>

#include "Alignment/MTDAlignment/interface/AlignableETLDee.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of modules and computes the surface from them
AlignableETLDee::AlignableETLDee(const std::vector<AlignableETLServiceHybrid*>& etlServiceHybrids)
    : AlignableComposite(etlServiceHybrids[0]->id(), align::AlignableETLDee) {
  theETLServiceHybrids.insert(theETLServiceHybrids.end(), etlServiceHybrids.begin(), etlServiceHybrids.end());

  // maintain also list of components
  for (const auto& sh : etlServiceHybrids) {
    const auto mother = sh->mother();
    this->addComponent(sh);  // components will be deleted by dtor of AlignableComposite
    sh->setMother(mother);   // restore previous behaviour where mother is not set
  }
  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return AlignableETLDee station at given index
AlignableETLServiceHybrid& AlignableETLDee::serviceHybrid(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Service Hybrid index (" << i << ") out of range";

  return *theETLServiceHybrids[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableETLDee::computeSurface() { return AlignableSurface(computePosition(), computeOrientation()); }

/// Compute average z position from all components (x and y forced to 0)
AlignableETLDee::PositionType AlignableETLDee::computePosition() {
  float xx = 0.;
  float yy = 0.;
  float zz = 0.;

  for (std::vector<AlignableETLServiceHybrid*>::iterator imodule = theETLServiceHybrids.begin();
       imodule != theETLServiceHybrids.end();
       imodule++) {
    xx += (*imodule)->globalPosition().x();
    yy += (*imodule)->globalPosition().y();
    zz += (*imodule)->globalPosition().z();
  }
  xx /= static_cast<float>(theETLServiceHybrids.size());
  yy /= static_cast<float>(theETLServiceHybrids.size());
  zz /= static_cast<float>(theETLServiceHybrids.size());

  return PositionType(xx, yy, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableETLDee::RotationType AlignableETLDee::computeOrientation() { return RotationType(); }

/// Output Half Barrel information
std::ostream& operator<<(std::ostream& os, const AlignableETLDee& b) {
  os << "This ETL Dee contains " << b.theETLServiceHybrids.size() << " ETL Service Hybrid" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole Half Barrel structure
void AlignableETLDee::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableETLServiceHybrid*>::const_iterator ish = theETLServiceHybrids.begin();
       ish != theETLServiceHybrids.end();
       ish++)
    (*ish)->dump();
}

//__________________________________________________________________________________________________

Alignments* AlignableETLDee::alignments(void) const {
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

AlignmentErrorsExtended* AlignableETLDee::alignmentErrors(void) const {
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

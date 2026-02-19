/** \file
 *
 *  $Date: 2024/12/10 16:36:41 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include <memory>

#include "Alignment/MTDAlignment/interface/AlignableETLModule.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// The constructor simply copies the vector of modules and computes the surface from them
AlignableETLModule::AlignableETLModule(const std::vector<AlignableETLSensor*>& etlSensors)
    : AlignableComposite(etlSensors[0]->id(), align::AlignableETLModule) {
  theETLSensors.insert(theETLSensors.end(), etlSensors.begin(), etlSensors.end());

  // maintain also list of components
  for (const auto& sensor : etlSensors) {
    const auto mother = sensor->mother();
    this->addComponent(sensor);  // components will be deleted by dtor of AlignableComposite
    sensor->setMother(mother);   // restore previous behaviour where mother is not set
  }
  setSurface(computeSurface());
  compConstraintType_ = Alignable::CompConstraintType::POSITION_Z;
}

/// Return AlignableETLModule station at given index
AlignableETLSensor& AlignableETLModule::sensor(int i) {
  if (i >= size())
    throw cms::Exception("LogicError") << "Module index (" << i << ") out of range";

  return *theETLSensors[i];
}

/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableETLModule::computeSurface() {
  return AlignableSurface(computePosition(), computeOrientation());
}

/// Compute average z position from all components (x and y forced to 0)
AlignableETLModule::PositionType AlignableETLModule::computePosition() {
  float xx = 0.;
  float yy = 0.;
  float zz = 0.;

  for (std::vector<AlignableETLSensor*>::iterator isensor = theETLSensors.begin(); isensor != theETLSensors.end();
       isensor++) {
    xx += (*isensor)->globalPosition().x();
    yy += (*isensor)->globalPosition().y();
    zz += (*isensor)->globalPosition().z();
  }
  xx /= static_cast<float>(theETLSensors.size());
  yy /= static_cast<float>(theETLSensors.size());
  zz /= static_cast<float>(theETLSensors.size());

  return PositionType(xx, yy, zz);
}

/// Just initialize to default given by default constructor of a RotationType
AlignableETLModule::RotationType AlignableETLModule::computeOrientation() { return RotationType(); }

/// Output Half Barrel information
std::ostream& operator<<(std::ostream& os, const AlignableETLModule& b) {
  os << "This ETL Module contains " << b.theETLSensors.size() << " ETL sensors" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," << b.globalPosition().perp() << ","
     << b.globalPosition().z();
  os << "),  orientation:" << std::endl << b.globalRotation() << std::endl;
  return os;
}

/// Recursive printout of whole structure
void AlignableETLModule::dump(void) const {
  edm::LogInfo("AlignableDump") << (*this);
  for (std::vector<AlignableETLSensor*>::const_iterator isensor = theETLSensors.begin(); isensor != theETLSensors.end();
       isensor++)
    (*isensor)->dump();
}

//__________________________________________________________________________________________________

Alignments* AlignableETLModule::alignments(void) const {
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

AlignmentErrorsExtended* AlignableETLModule::alignmentErrors(void) const {
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

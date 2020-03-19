#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//__________________________________________________________________________________________________
AlignableDet::AlignableDet(const GeomDet* geomDet, bool addComponents)
    : AlignableComposite(geomDet), theAlignmentPositionError(nullptr) {
  // ensure that the surface is not constrained to the average position of the
  // components:
  compConstraintType_ = Alignable::CompConstraintType::NONE;

  if (geomDet->alignmentPositionError()) {
    // false: do not propagate APE to (anyway not yet existing) daughters
    this->setAlignmentPositionError(*(geomDet->alignmentPositionError()), false);
  }

  if (addComponents) {
    if (geomDet->components().empty()) {  // Is a DetUnit
      throw cms::Exception("BadHierarchy") << "[AlignableDet] GeomDet with DetId " << geomDet->geographicalId().rawId()
                                           << " has no components, use AlignableDetUnit.\n";
    } else {  // Push back all components
      const std::vector<const GeomDet*>& geomDets = geomDet->components();
      for (std::vector<const GeomDet*>::const_iterator idet = geomDets.begin(); idet != geomDets.end(); ++idet) {
        const GeomDetUnit* unit = dynamic_cast<const GeomDetUnit*>(*idet);
        if (!unit) {
          throw cms::Exception("BadHierarchy")
              << "[AlignableDet] component not GeomDetUnit, call with addComponents==false"
              << " and build hierarchy yourself.\n";  // e.g. AlignableDTChamber
        }
        this->addComponent(new AlignableDetUnit(unit));
      }
    }
    // Ensure that the surface is not screwed up by addComponent, it must stay the GeomDet's one:
    theSurface = AlignableSurface(geomDet->surface());
  }  // end addComponents
}

//__________________________________________________________________________________________________
AlignableDet::~AlignableDet() { delete theAlignmentPositionError; }

//______________________________________________________________________________
void AlignableDet::update(const GeomDet* geomDet, bool updateComponents) {
  AlignableComposite::update(geomDet);

  if (geomDet->alignmentPositionError()) {
    // false: do not propagate APE to daughters (done by their update functions)
    this->setAlignmentPositionError(*(geomDet->alignmentPositionError()), false);
  }

  if (updateComponents) {
    if (geomDet->components().empty()) {  // Is a DetUnit
      throw cms::Exception("BadHierarchy") << "[AlignableDet] GeomDet with DetId " << geomDet->geographicalId().rawId()
                                           << " has no components, use AlignableDetUnit.\n";
    } else {  // Push back all components
      const auto& geomDets = geomDet->components();
      for (const auto& idet : geomDets) {
        auto unit = dynamic_cast<const GeomDetUnit*>(idet);
        if (!unit) {
          throw cms::Exception("BadHierarchy") << "[AlignableDet] component not GeomDetUnit, call with "
                                               << "updateComponents==false and build hierarchy yourself.\n";
          // -> e.g. AlignableDTChamber
        }

        const auto components = this->components();
        auto comp = std::find_if(components.begin(), components.end(), [&unit](const auto& c) {
          return c->id() == unit->geographicalId().rawId();
        });

        if (comp != components.end()) {
          auto aliDetUnit = dynamic_cast<AlignableDetUnit*>(*comp);
          if (aliDetUnit) {
            aliDetUnit->update(unit);
          } else {
            throw cms::Exception("LogicError") << "[AlignableDet::update] cast to 'AlignableDetUnit*' failed "
                                               << "while it should not\n";
          }
        } else {
          throw cms::Exception("GeometryMismatch")
              << "[AlignableDet::update] GeomDet with DetId " << unit->geographicalId().rawId()
              << " not found in current geometry.\n";
        }
      }
    }
    // Ensure that the surface is not screwed up by update of components, it must stay the GeomDet's one:
    theSurface = AlignableSurface(geomDet->surface());
  }  // end updateComponents
}

//__________________________________________________________________________________________________
void AlignableDet::setAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown) {
  if (!theAlignmentPositionError)
    theAlignmentPositionError = new AlignmentPositionError(ape);
  else
    *theAlignmentPositionError = ape;

  this->AlignableComposite::setAlignmentPositionError(ape, propagateDown);
}

//__________________________________________________________________________________________________
void AlignableDet::addAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown) {
  if (!theAlignmentPositionError) {
    theAlignmentPositionError = new AlignmentPositionError(ape);
  } else {
    *theAlignmentPositionError += ape;
  }

  this->AlignableComposite::addAlignmentPositionError(ape, propagateDown);
}

//__________________________________________________________________________________________________
void AlignableDet::addAlignmentPositionErrorFromRotation(const RotationType& rot, bool propagateDown) {
  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  GlobalVector localPositionVector =
      surface().toGlobal(LocalVector(.5 * surface().width(), .5 * surface().length(), 0.));

  const LocalVector::BasicVectorType& lpvgf = localPositionVector.basicVector();
  GlobalVector gv(rot.multiplyInverse(lpvgf) - lpvgf);

  AlignmentPositionError ape(gv.x(), gv.y(), gv.z());
  this->addAlignmentPositionError(ape, propagateDown);

  this->AlignableComposite::addAlignmentPositionErrorFromRotation(rot, propagateDown);
}

//__________________________________________________________________________________________________
Alignments* AlignableDet::alignments() const {
  Alignments* m_alignments = new Alignments();
  RotationType rot(this->globalRotation());

  // Get position, rotation, detId
  CLHEP::Hep3Vector clhepVector(globalPosition().x(), globalPosition().y(), globalPosition().z());
  CLHEP::HepRotation clhepRotation(
      CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()));
  uint32_t detId = this->geomDetId().rawId();

  AlignTransform transform(clhepVector, clhepRotation, detId);

  // Add to alignments container
  m_alignments->m_align.push_back(transform);

  // Add those from components
  Alignments* compAlignments = this->AlignableComposite::alignments();
  std::copy(compAlignments->m_align.begin(), compAlignments->m_align.end(), std::back_inserter(m_alignments->m_align));
  delete compAlignments;

  return m_alignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableDet::alignmentErrors(void) const {
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add associated alignment position error
  uint32_t detId = this->geomDetId().rawId();
  CLHEP::HepSymMatrix clhepSymMatrix(6, 0);
  if (theAlignmentPositionError)  // Might not be set
    clhepSymMatrix = asHepMatrix(theAlignmentPositionError->globalError().matrix());
  AlignTransformErrorExtended transformError(clhepSymMatrix, detId);
  m_alignmentErrors->m_alignError.push_back(transformError);

  // Add those from components
  AlignmentErrorsExtended* compAlignmentErrs = this->AlignableComposite::alignmentErrors();
  std::copy(compAlignmentErrs->m_alignError.begin(),
            compAlignmentErrs->m_alignError.end(),
            std::back_inserter(m_alignmentErrors->m_alignError));
  delete compAlignmentErrs;

  return m_alignmentErrors;
}

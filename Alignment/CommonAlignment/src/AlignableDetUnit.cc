#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/AlignmentPositionError.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//__________________________________________________________________________________________________
AlignableDetUnit::AlignableDetUnit(const GeomDetUnit* geomDetUnit)
    :  // rely on non-NULL pointer!
      Alignable(geomDetUnit->geographicalId().rawId(), geomDetUnit->surface()),
      theAlignmentPositionError(nullptr),
      theSurfaceDeformation(nullptr),
      theCachedSurfaceDeformation(nullptr) {
  if (geomDetUnit->alignmentPositionError()) {  // take over APE from geometry
    // 2nd argument w/o effect:
    this->setAlignmentPositionError(*(geomDetUnit->alignmentPositionError()), false);
  }

  if (geomDetUnit->surfaceDeformation()) {  // take over surface modification
    // 2nd argument w/o effect:
    this->setSurfaceDeformation(geomDetUnit->surfaceDeformation(), false);
  }

  theDeepComponents.push_back(this);
}

//__________________________________________________________________________________________________
AlignableDetUnit::~AlignableDetUnit() {
  delete theAlignmentPositionError;
  delete theSurfaceDeformation;
  delete theCachedSurfaceDeformation;
  for (auto surface : surfaceDeformationsCache_)
    delete surface.second;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::update(const GeomDetUnit* geomDetUnit) {
  if (!geomDetUnit) {
    throw cms::Exception("Alignment") << "@SUB=AlignableDetUnit::update\n"
                                      << "Trying to update with GeomDetUnit* pointing to 'nullptr'.";
  }

  Alignable::update(geomDetUnit->geographicalId().rawId(), geomDetUnit->surface());

  if (geomDetUnit->alignmentPositionError()) {  // take over APE from geometry
    // 2nd argument w/o effect:
    this->setAlignmentPositionError(*(geomDetUnit->alignmentPositionError()), false);
  }

  if (geomDetUnit->surfaceDeformation()) {  // take over surface modification
    // 2nd argument w/o effect:
    this->setSurfaceDeformation(geomDetUnit->surfaceDeformation(), false);
  }
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addComponent(Alignable* /*unused*/) {
  throw cms::Exception("LogicError") << "AlignableDetUnit cannot have components, but try to add one!";
}

//__________________________________________________________________________________________________
void AlignableDetUnit::move(const GlobalVector& displacement) {
  theSurface.move(displacement);
  this->addDisplacement(displacement);
}

//__________________________________________________________________________________________________
void AlignableDetUnit::rotateInGlobalFrame(const RotationType& rotation) {
  theSurface.rotate(rotation);
  this->addRotation(rotation);
}

//__________________________________________________________________________________________________
void AlignableDetUnit::setAlignmentPositionError(const AlignmentPositionError& ape, bool /*propagateDown*/) {
  if (!theAlignmentPositionError)
    theAlignmentPositionError = new AlignmentPositionError(ape);
  else
    *theAlignmentPositionError = ape;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown) {
  if (!theAlignmentPositionError)
    this->setAlignmentPositionError(ape, propagateDown);  // 2nd argument w/o effect
  else
    *theAlignmentPositionError += ape;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromRotation(const RotationType& rot, bool propagateDown) {
  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  GlobalVector localPositionVector =
      surface().toGlobal(LocalVector(.5 * surface().width(), .5 * surface().length(), 0.));

  const LocalVector::BasicVectorType& lpvgf = localPositionVector.basicVector();
  GlobalVector gv(rot.multiplyInverse(lpvgf) - lpvgf);

  AlignmentPositionError ape(gv.x(), gv.y(), gv.z());
  this->addAlignmentPositionError(ape, propagateDown);  // 2nd argument w/o effect
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromLocalRotation(const RotationType& rot, bool propagateDown) {
  RotationType globalRot = globalRotation().multiplyInverse(rot * globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot, propagateDown);  // 2nd argument w/o effect
}

//__________________________________________________________________________________________________
void AlignableDetUnit::setSurfaceDeformation(const SurfaceDeformation* deformation, bool /* propagateDown */) {
  delete theSurfaceDeformation;  // OK for zero pointers
  if (deformation) {
    theSurfaceDeformation = deformation->clone();
  } else {
    theSurfaceDeformation = nullptr;
  }
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addSurfaceDeformation(const SurfaceDeformation* deformation, bool propagateDown) {
  if (!deformation) {
    // nothing to do
  } else if (!theSurfaceDeformation) {
    this->setSurfaceDeformation(deformation, propagateDown);  // fine since no components
  } else if (!theSurfaceDeformation->add(*deformation)) {
    edm::LogError("Alignment") << "@SUB=AlignableDetUnit::addSurfaceDeformation"
                               << "Cannot add deformation type " << deformation->type() << " to type "
                               << theSurfaceDeformation->type() << ", so erase deformation information.";
    delete theSurfaceDeformation;
    theSurfaceDeformation = nullptr;
  }
}

//__________________________________________________________________________________________________
void AlignableDetUnit::dump() const {
  std::ostringstream parameters;
  if (theSurfaceDeformation) {
    parameters << "    surface deformation parameters:";
    for (const auto& param : theSurfaceDeformation->parameters()) {
      parameters << " " << param;
    }
  } else {
    parameters << "    no surface deformation parameters";
  }

  edm::LogInfo("AlignableDump") << " AlignableDetUnit has position = " << this->globalPosition()
                                << ", orientation:" << std::endl
                                << this->globalRotation() << std::endl
                                << " total displacement and rotation: " << this->displacement() << std::endl
                                << this->rotation() << "\n"
                                << parameters.str();
}

//__________________________________________________________________________________________________
Alignments* AlignableDetUnit::alignments() const {
  Alignments* m_alignments = new Alignments();
  RotationType rot(this->globalRotation());

  // Get alignments (position, rotation, detId)
  CLHEP::Hep3Vector clhepVector(globalPosition().x(), globalPosition().y(), globalPosition().z());
  CLHEP::HepRotation clhepRotation(
      CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()));
  uint32_t detId = this->geomDetId().rawId();

  AlignTransform transform(clhepVector, clhepRotation, detId);

  // Add to alignments container
  m_alignments->m_align.push_back(transform);

  return m_alignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableDetUnit::alignmentErrors() const {
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  uint32_t detId = this->geomDetId().rawId();

  CLHEP::HepSymMatrix clhepSymMatrix(6, 0);
  if (theAlignmentPositionError)  // Might not be set
    clhepSymMatrix = asHepMatrix(theAlignmentPositionError->globalError().matrix());

  AlignTransformErrorExtended transformError(clhepSymMatrix, detId);

  m_alignmentErrors->m_alignError.push_back(transformError);

  return m_alignmentErrors;
}

//__________________________________________________________________________________________________
int AlignableDetUnit::surfaceDeformationIdPairs(std::vector<std::pair<int, SurfaceDeformation*> >& result) const {
  if (theSurfaceDeformation) {
    result.push_back(std::pair<int, SurfaceDeformation*>(this->geomDetId().rawId(), theSurfaceDeformation));
    return 1;
  }

  return 0;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::cacheTransformation() {
  theCachedSurface = theSurface;
  theCachedDisplacement = theDisplacement;
  theCachedRotation = theRotation;

  if (theCachedSurfaceDeformation) {
    delete theCachedSurfaceDeformation;
    theCachedSurfaceDeformation = nullptr;
  }

  if (theSurfaceDeformation)
    theCachedSurfaceDeformation = theSurfaceDeformation->clone();
}

//__________________________________________________________________________________________________
void AlignableDetUnit::cacheTransformation(const align::RunNumber& run) {
  surfacesCache_[run] = theSurface;
  displacementsCache_[run] = theDisplacement;
  rotationsCache_[run] = theRotation;

  auto existingCache = surfaceDeformationsCache_.find(run);
  if (existingCache != surfaceDeformationsCache_.end()) {
    delete existingCache->second;
    existingCache->second = nullptr;
  }

  if (theSurfaceDeformation) {
    surfaceDeformationsCache_[run] = theSurfaceDeformation->clone();
  }
}

//__________________________________________________________________________________________________
void AlignableDetUnit::restoreCachedTransformation() {
  theSurface = theCachedSurface;
  theDisplacement = theCachedDisplacement;
  theRotation = theCachedRotation;

  if (theSurfaceDeformation) {
    delete theSurfaceDeformation;
    theSurfaceDeformation = nullptr;
  }

  if (theCachedSurfaceDeformation) {
    this->setSurfaceDeformation(theCachedSurfaceDeformation, false);
  }
}

//__________________________________________________________________________________________________
void AlignableDetUnit::restoreCachedTransformation(const align::RunNumber& run) {
  if (surfacesCache_.find(run) == surfacesCache_.end()) {
    throw cms::Exception("Alignment") << "@SUB=Alignable::restoreCachedTransformation\n"
                                      << "Trying to restore cached transformation for a run (" << run
                                      << ") that has not been cached.";
  } else {
    theSurface = surfacesCache_[run];
    theDisplacement = displacementsCache_[run];
    theRotation = rotationsCache_[run];

    if (theSurfaceDeformation) {
      delete theSurfaceDeformation;
      theSurfaceDeformation = nullptr;
    }

    if (surfaceDeformationsCache_[run]) {
      this->setSurfaceDeformation(surfaceDeformationsCache_[run], false);
    }
  }
}

//______________________________________________________________________________
const align::Alignables AlignableDetUnit::emptyComponents_{};

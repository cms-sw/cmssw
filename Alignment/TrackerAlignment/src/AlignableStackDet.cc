/* 
 *  $Date: 2011/09/05 16:59:07 $
 *  $Revision: 1.9 $
 */

#include "Alignment/CommonAlignment/interface/AlignableSurface.h"
#include "Alignment/TrackerAlignment/interface/AlignableStackDet.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include <cmath>

AlignableStackDet::AlignableStackDet(const StackGeomDet* stackedDet)
    : AlignableDet(stackedDet, true),  // true: adding DetUnits
      theLowerDetSurface(stackedDet->lowerDet()->surface()) {
  // check order lower/upper
  const Alignables units(this->components());
  if (units.size() != 2 || stackedDet->lowerDet()->geographicalId() != units[0]->geomDetId() ||
      stackedDet->upperDet()->geographicalId() != units[1]->geomDetId()) {
    throw cms::Exception("LogicError") << "[AlignableStackDet] "
                                       << "Either != 2 components or "
                                       << "upper/lower in wrong order for consistifyAlignments.";
  }
}

//__________________________________________________________________________________________________
Alignments* AlignableStackDet::alignments() const {
  const_cast<AlignableStackDet*>(this)->consistifyAlignments();
  return this->AlignableDet::alignments();
}

//__________________________________________________________________________________________________
void AlignableStackDet::consistifyAlignments() {
  // Now we have all to calculate new position and rotation via PlaneBuilderForGluedDet.
  const PositionType oldPos(theSurface.position());  // From old surface for keeping...
  const RotationType oldRot(theSurface.rotation());  // ...track of changes.

  // The plane is *not* built in the middle, but on the Lower surface
  // see usage in  Geometry/TrackerGeometryBuilder/src/TrackerGeomBuilderFromGeometricDet.cc
  theSurface = AlignableSurface(theLowerDetSurface);

  // But do not forget to keep track of movements/rotations:
  const GlobalVector movement(theSurface.position().basicVector() - oldPos.basicVector());
  // Seems to be correct down to delta angles 4.*1e-8:
  const RotationType rotation(oldRot.multiplyInverse(theSurface.rotation()));
  this->addDisplacement(movement);
  this->addRotation(rotation);
}

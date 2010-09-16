/* 
 *  $Date: 2009/04/16 16:53:07 $
 *  $Revision: 1.7 $
 */

#include "Alignment/TrackerAlignment/interface/AlignableSiStripDet.h"
 
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"

#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderForGluedDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <math.h>

AlignableSiStripDet::AlignableSiStripDet(const GluedGeomDet *gluedDet) 
  : AlignableDet(gluedDet, true), // true: adding DetUnits
    theMonoBounds  (gluedDet->monoDet()  ->surface().bounds().clone()),
    theStereoBounds(gluedDet->stereoDet()->surface().bounds().clone()),
    theMonoType  (static_cast<const StripGeomDetUnit*>(gluedDet->monoDet())  ->specificType()),
    theStereoType(static_cast<const StripGeomDetUnit*>(gluedDet->stereoDet())->specificType())
{
  // It is not allowed to store a pointer to GeomDet within objects with a life time
  // longer than an Event: 
  // GeomDet comes from TrackerGeometry that is created from GeometricDet that depends on
  // IdealGeometryRecord from EventSetup, so it could change in next event!
  //  ==> Need to store directly what I need from it. 
  // Unfortunately the current way with references for the type does not solve that,
  // either. But currently no way out, see header file.

  // check order mono/stereo
  const Alignables units(this->components());
  if (units.size() != 2
      || gluedDet->monoDet()->geographicalId() != units[0]->geomDetId()
      || gluedDet->stereoDet()->geographicalId() != units[1]->geomDetId()) {
    throw cms::Exception("LogicError")
      << "[AlignableSiStripDet] " << "Either != 2 components or "
      << "mono/stereo in wrong order for consistifyAlignments.";
  }
}

//__________________________________________________________________________________________________
AlignableSiStripDet::~AlignableSiStripDet()
{
  delete theMonoBounds;
  delete theStereoBounds;
}

//__________________________________________________________________________________________________
Alignments* AlignableSiStripDet::alignments() const
{
  const_cast<AlignableSiStripDet*>(this)->consistifyAlignments();

  return this->AlignableDet::alignments();
}

//__________________________________________________________________________________________________
void AlignableSiStripDet::consistifyAlignments()
{
  // make alignments consistent with daughters, calling method from geometry

  // The aim of all this gymnastics is to have the alignments calculated the same way as
  // in PlaneBuilderForGluedDet::plane(const std::vector<const GeomDetUnit*> &detComps);
  // 
  // So we take the (new) position and orientation from the AligableDetUnits,
  // but bounds and GeomDetType from original GeomDetUnits to create new a new glued
  // surface.

  const PositionType oldPos(theSurface.position()); // From old surface for keeping...
  const RotationType oldRot(theSurface.rotation()); // ...track of changes.

  const Alignables aliUnits(this->components()); // order mono==0, stereo==1 checked in ctr.
  
  BoundPlane::BoundPlanePointer monoPlane
    = BoundPlane::build(aliUnits[0]->globalPosition(), aliUnits[0]->globalRotation(),
			*theMonoBounds);

  BoundPlane::BoundPlanePointer stereoPlane
    = BoundPlane::build(aliUnits[1]->globalPosition(), aliUnits[1]->globalRotation(),
			*theStereoBounds);
  
  Surface::PositionType::BasicVectorType posSum = monoPlane->position().basicVector();
  posSum += stereoPlane->position().basicVector();
  Surface::PositionType meanPos(posSum/(float)aliUnits.size());
  Surface::RotationType rotation =  monoPlane->rotation();
  
  BoundPlane::BoundPlanePointer tmpPlane = BoundPlane::build(meanPos, rotation, OpenBounds());

  const MediumProperties* mp = monoPlane->mediumProperties();
  MediumProperties newmp(0,0);
  if (mp != 0) newmp = MediumProperties(mp->radLen()*2.0,mp->xi()*2.0);

  std::vector<GlobalPoint> corners;
  std::vector<GlobalPoint> monoDC = BoundingBox().corners(*monoPlane);
  corners.insert(corners.end(), monoDC.begin(), monoDC.end());
  std::vector<GlobalPoint> stereoDC = BoundingBox().corners(*stereoPlane);
  corners.insert(corners.end(), stereoDC.begin(), stereoDC.end());

  float xmin(0), xmax(0), ymin(0), ymax(0), zmin(0), zmax(0);
  for (std::vector<GlobalPoint>::const_iterator i=corners.begin();
       i!=corners.end();
       ++i) {
    LocalPoint p = tmpPlane->toLocal(*i);
    if (p.x() < xmin) xmin = p.x();
    if (p.x() > xmax) xmax = p.x();
    if (p.y() < ymin) ymin = p.y();
    if (p.y() > ymax) ymax = p.y();
    if (p.z() < zmin) zmin = p.z();
    if (p.z() > zmax) zmax = p.z();
  }
 
  LocalVector localOffset((xmin+xmax)/2., (ymin+ymax)/2., (zmin+zmax)/2.);
  GlobalVector offset(tmpPlane->toGlobal(localOffset));

  theSurface = AlignableSurface(BoundPlane(meanPos+offset, rotation, 
					   RectangularPlaneBounds((xmax-xmin)/2, (ymax-ymin)/2, (zmax-zmin)/2),
					   &newmp));
  
  // But do not forget to keep track of movements/rotations:
  const GlobalVector theMovement(theSurface.position().basicVector() - oldPos.basicVector());
  // Seems to be correct down to delta angles 4.*1e-8:
  const RotationType theRotation(oldRot.multiplyInverse(theSurface.rotation()));
  this->addDisplacement(theMovement);
  this->addRotation(theRotation);
  
//   this->dumpCompareEuler(oldRot, theSurface.rotation());

//   if (movement.mag2()) { // > 1.e-10) { 
//     edm::LogWarning("Alignment") << "@SUB=consistifyAlignments" 
//  				 << "Delta: " << movement.x() << " " << movement.y() << " " << movement.z()
// 				 << "\nPos: " << oldPos.perp() << " " << oldPos.phi() << " " << oldPos.z();
//   }
}

// #include "CLHEP/Vector/EulerAngles.h"
// #include "CLHEP/Vector/Rotation.h"
// //__________________________________________________________________________________________________
// void AlignableSiStripDet::dumpCompareEuler(const RotationType &oldRot,
// 					   const RotationType &newRot) const
// {
//   // 
//   const HepRotation oldClhep(HepRep3x3(oldRot.xx(), oldRot.xy(), oldRot.xz(),
// 				       oldRot.yx(), oldRot.yy(), oldRot.yz(),
// 				       oldRot.zx(), oldRot.zy(), oldRot.zz()));
//
//   const HepRotation newClhep(HepRep3x3(newRot.xx(), newRot.xy(), newRot.xz(),
// 				       newRot.yx(), newRot.yy(), newRot.yz(),
// 				       newRot.zx(), newRot.zy(), newRot.zz()));
//
//   const RotationType rotationGlob(oldRot.multiplyInverse(newRot));
//   const RotationType rotation(theSurface.toLocal(rotationGlob)); // not 100% correct: new global...
//   const HepRotation diff(HepRep3x3(rotation.xx(), rotation.xy(), rotation.xz(),
// 				       rotation.yx(), rotation.yy(), rotation.yz(),
// 				       rotation.zx(), rotation.zy(), rotation.zz()));
//
//   edm::LogWarning("Alignment") << "@SUB=dumpCompareEuler" 
//  			       << "oldEuler " << oldClhep.eulerAngles()
//  			       << "\nnewEuler " << newClhep.eulerAngles()
// 			       << "\n diff_euler " << diff.eulerAngles()
// 			       << "\n diff_diag (" << diff.xx() << ", " << diff.yy() 
// 			       <<         ", " << diff.zz() << ")";
// }

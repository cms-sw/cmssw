/* 
 *  $Date: 2008/05/02 07:54:22 $
 *  $Revision: 1.1 $
 */

#include "Alignment/TrackerAlignment/interface/AlignableSiStripDet.h"
 
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"

#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderForGluedDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <math.h>

AlignableSiStripDet::AlignableSiStripDet(const GluedGeomDet *gluedDet) 
  : AlignableDet(gluedDet, true), // true: adding DetUnits
    theMonoBounds  (gluedDet->monoDet()  ->surface().bounds()), // .clone()
    theStereoBounds(gluedDet->stereoDet()->surface().bounds()), // .clone()
    theMonoType  (static_cast<const StripGeomDetUnit*>(gluedDet->monoDet())  ->specificType()),
    theStereoType(static_cast<const StripGeomDetUnit*>(gluedDet->stereoDet())->specificType())
{
  // It is not allowed to store a pointer to GeomDet within objects with a life time
  // longer than an Event: 
  // GeomDet comes from TrackerGeometry that is created from GeometricDet that depends on
  // IdealGeometryRecord from EventSetup, so it could change in next event!
  //  ==> Need to store directly what I need from it. 
  // Unfortunately the current way with references for bounds and type does not solve that,
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
Alignments* AlignableSiStripDet::alignments() const
{
  const_cast<AlignableSiStripDet*>(this)->consistifyAlignments();

  return this->AlignableDet::alignments();
}

//__________________________________________________________________________________________________
AlignmentErrors* AlignableSiStripDet::alignmentErrors() const
{
  const_cast<AlignableSiStripDet*>(this)->consistifyAlignmentErrors();

  return this->AlignableDet::alignmentErrors();
}

//__________________________________________________________________________________________________
void AlignableSiStripDet::consistifyAlignments()
{
  // make alignments consistent with daughters, calling method from geometry

  // The aim of all this gymnastics is to have the alignments calculated by
  // PlaneBuilderForGluedDet::plane(const std::vector<const GeomDetUnit*> &detComps);
  // 
  // So we take the (new) position and orientation from the AligableDetUnits,
  // but bounds and GeomDetType from original GeomDetUnits to create new GeomDetUnits
  // that are passed to that routine.

  const Alignables aliUnits(this->components()); // order mono==0, stereo==1 checked in ctr.

  BoundPlane::BoundPlanePointer monoPlane
    = BoundPlane::build(aliUnits[0]->globalPosition(), aliUnits[0]->globalRotation(),
			theMonoBounds);
  // Fortunately we do not seem to need a GeometricDet pointer and can use 0:
  const StripGeomDetUnit monoDet(&(*monoPlane), &theMonoType, 0);

  BoundPlane::BoundPlanePointer stereoPlane
    = BoundPlane::build(aliUnits[1]->globalPosition(), aliUnits[1]->globalRotation(),
			theStereoBounds);
  // Fortunately we do not seem to need a GeometricDet pointer and can use 0:
  const StripGeomDetUnit stereoDet(&(*stereoPlane), &theStereoType, 0);

  std::vector<const GeomDetUnit*> detComps;
  detComps.push_back(&monoDet);   // order mono first, stereo second should be as in...
  detComps.push_back(&stereoDet); // ...TrackerGeomBuilderFromGeometricDet::buildGeomDet

  // Now we have all to calculate new position (and rotation) via PlaneBuilderForGluedDet.
  const PositionType oldPos = theSurface.position(); // From old surface for tracking of movement!
  PlaneBuilderForGluedDet planeBuilder;
  theSurface = AlignableSurface(*planeBuilder.plane(detComps));

  // But do not forget to keep track of movements/rotations:
  const GlobalVector movement(theSurface.position().basicVector() - oldPos.basicVector());
  const RotationType rotation; // FIXME: Calculate!! 
  this->addDisplacement(movement);
  this->addRotation(rotation);

//   if (movement.mag2()) { // > 1.e-10) { 
//     edm::LogWarning("Alignment") << "@SUB=consistifyAlignments" 
//  				 << "Delta: " << movement.x() << " " << movement.y() << " " << movement.z()
// 				 << "\nPos: " << oldPos.perp() << " " << oldPos.phi() << " " << oldPos.z();
//   }

}


//__________________________________________________________________________________________________
void AlignableSiStripDet::consistifyAlignmentErrors()
{
  // make alignment errors consistent with daughters

  AlignmentErrors *oldErrs = this->AlignableDet::alignmentErrors();

  const Alignables units(this->components()); // order mono==0, stereo==1 does not matter here

  const AlignTransformError &gluedErr  = this->errorFromId(oldErrs->m_alignError,
							   this->geomDetId());
  const AlignTransformError &monoErr   = this->errorFromId(oldErrs->m_alignError,
							   units[0]->geomDetId());
  const AlignTransformError &stereoErr = this->errorFromId(oldErrs->m_alignError,
							   units[1]->geomDetId());
  const GlobalError errGlued (gluedErr.matrix());
  const GlobalError errMono  (monoErr.matrix());
  const GlobalError errStereo(stereoErr.matrix());

  //  const GlobalError newErrGlued((errMono + errStereo - errGlued).matrix_new() /= 4.);
  // The above line would be error propagation assuming:
  //   - Glued position is just the mean of its components.
  //   - Components APE is square sum of what has been set to glued and to components itself.
  // But this can be too small, e.g. 
  // - Simply by factor sqrt(4)=2 smaller than the old 'errGlued' in case APE (from position)
  //   was only set to glued (and 1-to-1 propagated to components).
  // - Factor sqrt(2) smaller than components APE in case APE (from position) was only
  //   directly applied to both components.
  //
  // So I choose the max of all three, ignoring correlations (which we do not have?), sigh!
  // And in this way it is safe against repetetive calls to this method!
  double maxX2 = (errMono.cxx() > errStereo.cxx() ? errMono.cxx() : errStereo.cxx());
  maxX2 = (maxX2 > errGlued.cxx() ? maxX2 : errGlued.cxx());
  double maxY2 = (errMono.cyy() > errStereo.cyy() ? errMono.cyy() : errStereo.cyy());
  maxY2 = (maxY2 > errGlued.cyy() ? maxY2 : errGlued.cyy());
  double maxZ2 = (errMono.czz() > errStereo.czz() ? errMono.czz() : errStereo.czz());
  maxZ2 = (maxZ2 > errGlued.czz() ? maxZ2 : errGlued.czz());
  const AlignmentPositionError newApeGlued(sqrt(maxX2), sqrt(maxY2), sqrt(maxZ2));

  // Now set new errors - and reset those of the components, since they get overwritten...
  this->setAlignmentPositionError(newApeGlued);
  units[0]->setAlignmentPositionError(AlignmentPositionError(errMono));
  units[1]->setAlignmentPositionError(AlignmentPositionError(errStereo));

//   edm::LogWarning("Alignment") << "@SUB=consistifyAlignmentErrors" 
// 			       << "End Id " << this->geomDetId();
//   AlignmentErrors *newErrs = this->AlignableDet::alignmentErrors();
//   this->dumpCompareAPE(oldErrs->m_alignError, newErrs->m_alignError);
//   delete newErrs;

  delete oldErrs;
}

//__________________________________________________________________________________________________
const AlignTransformError& 
AlignableSiStripDet::errorFromId(const std::vector<AlignTransformError> &trafoErrs,
				 align::ID id) const
{
  for (unsigned int i = 0; i < trafoErrs.size(); ++i) {
    if (trafoErrs[i].rawId() ==  id) return trafoErrs[i];
  }

  throw cms::Exception("Mismatch") << "[AlignableSiStripDet::indexFromId] "
				   << id << " not found.";

  return trafoErrs.front(); // never reached due to exception (but pleasing the compiler)
}


//__________________________________________________________________________________________________
void AlignableSiStripDet::dumpCompareAPE(const std::vector<AlignTransformError> &trafoErrs1,
					 const std::vector<AlignTransformError> &trafoErrs2) const
{

  for (unsigned int i = 0; i < trafoErrs1.size() && i < trafoErrs2.size(); ++i) {
    if (trafoErrs1[i].rawId() != trafoErrs2[i].rawId()) {
      // complain
      break;
    }
    const GlobalError globErr1(trafoErrs1[i].matrix());
    const GlobalError globErr2(trafoErrs2[i].matrix());
    edm::LogVerbatim("Alignment") << trafoErrs1[i].rawId() << " | " 
				  << globErr1.cxx() << " " 
				  << globErr1.cyy() << " " 
				  << globErr1.czz() << " | "
				  << globErr2.cxx() << " " 
				  << globErr2.cyy() << " " 
				  << globErr2.czz();

  }

}

#ifndef Alignment_TrackerAlignment_AlignableSiStripDet_H
#define Alignment_TrackerAlignment_AlignableSiStripDet_H

/** \class AlignableSiStripDet
 *  An alignable for GluedDets in Strip tracker, 
 *  taking care of consistency with AlignableDet components.
 *
 *  First implementation April/May 2008
 *  \author Gero Flucke, Hamburg University
 *  $Date: 2008/05/02 09:57:29 $
 *  $Revision: 1.2 $
 */
 
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include <vector>

class GluedGeomDet;
class AlignTransformError;
class Bounds;
class StripGeomDetType;

class AlignableSiStripDet: public AlignableDet {
 public:
  /// Constructor
  AlignableSiStripDet(const GluedGeomDet *geomDet);
  /// reduntantly make destructor virtual
  virtual ~AlignableSiStripDet() {}

  /// first consistify with component detunits, then call method from AlignableDet
  virtual Alignments* alignments() const;
  /// first consistify with component detunits, then call method from AlignableDet
  virtual AlignmentErrors* alignmentErrors() const;

 private:
  /// make alignments consistent with daughters
  void consistifyAlignments();
  /// make alignment errors consistent with daughters
  void consistifyAlignmentErrors();
  /// AlignTransformError with 'id'
  const AlignTransformError& errorFromId(const std::vector<AlignTransformError> &trafoErrs,
					 align::ID id) const;

//   void dumpCompareAPE(const std::vector<AlignTransformError> &trafoErrs1,
// 		      const std::vector<AlignTransformError> &trafoErrs2) const;
//   void dumpCompareEuler(const RotationType &oldRot, const RotationType &newRot) const;

  /// The following four members are needed to recalculate the surface in consistifyAlignments,
  /// to get rid of a GluedDet* which is disregarded since it could become an invalid pointer
  /// in the next event (theoretically...). But this solution is not better, the references  
  /// would become invalid together with the GeomDets they are taken from. For the Bounds I could
  /// use instead pointers and clone()/delete, but StripGeomDetType has neither clone() and nor a
  /// decent copy constructor. Sigh!
  const Bounds     &theMonoBounds;
  const Bounds     &theStereoBounds;
  StripGeomDetType &theMonoType;
  StripGeomDetType &theStereoType;
};

#endif

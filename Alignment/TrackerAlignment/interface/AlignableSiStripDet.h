#ifndef Alignment_TrackerAlignment_AlignableSiStripDet_H
#define Alignment_TrackerAlignment_AlignableSiStripDet_H

/** \class AlignableSiStripDet
 *  An alignable for GluedDets in Strip tracker, 
 *  taking care of consistency with AlignableDet components.
 *
 *  First implementation April/May 2008
 *  \author Gero Flucke, Hamburg University
 *  $Date: 2008/07/13 12:26:41 $
 *  $Revision: 1.4 $
 */
 
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include <vector>

class GluedGeomDet;
class AlignTransformErrorExtended;
class Bounds;
class StripGeomDetType;

class AlignableSiStripDet: public AlignableDet {
 public:
  /// Constructor
  AlignableSiStripDet(const GluedGeomDet *geomDet);
  /// reduntantly make destructor virtual
  ~AlignableSiStripDet() override;

  /// first consistify with component detunits, then call method from AlignableDet
  Alignments* alignments() const override;

 private:
  /// make alignments consistent with daughters
  void consistifyAlignments();

//   void dumpCompareEuler(const RotationType &oldRot, const RotationType &newRot) const;

  /// The following four members are needed to recalculate the surface in consistifyAlignments,
  /// to get rid of a GluedDet* which is disregarded since it could become an invalid pointer
  /// in the next event (theoretically...). But this solution is not better, the references for the
  /// types would become invalid together with the GeomDets they are taken from.
  /// StripGeomDetType has neither clone() and nor a decent copy constructor, so I cannot go the
  /// the same way as for the bounds. Sigh!
  const Bounds     *theMonoBounds;
  const Bounds     *theStereoBounds;
  const StripGeomDetType &theMonoType;
  const StripGeomDetType &theStereoType;
};

#endif

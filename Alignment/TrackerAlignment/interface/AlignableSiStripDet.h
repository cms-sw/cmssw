#ifndef Alignment_TrackerAlignment_AlignableSiStripDet_H
#define Alignment_TrackerAlignment_AlignableSiStripDet_H

/** \class AlignableSiStripDet
 *  An alignable for GluedDets in Strip tracker, 
 *  taking care of consistency with AlignableDet components.
 *
 *  First implementation April/May 2008
 *  $Date$
 *  $Revision$
 *  \author Gero Flucke, Hamburg University
 */
 
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include <vector>

class GluedGeomDet;
class AlignTransformError;

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

  void dumpCompareAPE(const std::vector<AlignTransformError> &trafoErrs1,
		      const std::vector<AlignTransformError> &trafoErrs2) const;

  const GluedGeomDet *theGluedDet; /// FIXME: see comment in constructor

//   Bounds theMonoBounds;
//   Bounds theStereoBounds;
//   StripGeomDetType theMonoType;
//   StripGeomDetType theStereoType;

};

#endif

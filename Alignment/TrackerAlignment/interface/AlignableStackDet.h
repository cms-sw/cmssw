#ifndef Alignment_TrackerAlignment_AlignableStackDet_H
#define Alignment_TrackerAlignment_AlignableStackDet_H

/** \class AlignableStackDet
 *  An alignable for StackDets in the Phase-2 Outer Tracker detector, 
 *  taking care of consistency with AlignableDet components.
 *
 *  First implementation March 2022
 *  \author Marco Musich, U. Pisa / INFN
 *  $Date: 2022/03/15 13:36:00 $
 *  $Revision: 1.0 $
 */

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"

class AlignTransformErrorExtended;
class Bounds;
class StripGeomDetType;

class AlignableStackDet : public AlignableDet {
public:
  /// Constructor
  AlignableStackDet(const StackGeomDet *geomDet);
  /// reduntantly make destructor virtual
  ~AlignableStackDet() override = default;

  /// first consistify with component detunits, then call method from AlignableDet
  Alignments *alignments() const override;

private:
  /// make alignments consistent with daughters
  void consistifyAlignments();
  const Plane theLowerDetSurface;
};

#endif

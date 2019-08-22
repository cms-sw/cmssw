#ifndef Alignment_MuonAlignment_AlignableDTChamber_H
#define Alignment_MuonAlignment_AlignableDTChamber_H

/** \class AlignableDTChamber
 *  The alignable muon DT chamber.
 *
 *  $Date: 2008/02/14 09:39:20 $
 *  $Revision: 1.12 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include <iosfwd>
#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// A muon DT Chamber( an AlignableDet )

class AlignableDTChamber : public AlignableDet {
public:
  friend std::ostream &operator<<(std::ostream &, const AlignableDTChamber &);

  /// Constructor
  AlignableDTChamber(const GeomDet *geomDet);
};

#endif  // ALIGNABLE_DT_CHAMBER_H

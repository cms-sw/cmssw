#ifndef Alignment_MuonAlignment_AlignableCSCChamber_H
#define Alignment_MuonAlignment_AlignableCSCChamber_H

/** \class AlignableCSCChamber
 *  The alignable muon CSC chamber.
 *
 *  $Date: 2008/03/26 21:58:50 $
 *  $Revision: 1.13 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 
 
#include <iosfwd> 
#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// A muon CSC Chamber( an AlignableDet )

class AlignableCSCChamber: public AlignableDet {
 public:
  friend std::ostream& operator<< (std::ostream&, const AlignableCSCChamber &);

  /// Constructor
  AlignableCSCChamber(const GeomDet *geomDet);
};

#endif  // ALIGNABLE_CSC_CHAMBER_H

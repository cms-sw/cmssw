#ifndef Alignment_MuonAlignment_AlignableDTSuperLayer_H
#define Alignment_MuonAlignment_AlignableDTSuperLayer_H

/** \class AlignableDTSuperLayer
 *  The alignable muon DT superlayer.
 *
 *  $Date: 2008/03/26 21:59:00 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski - Texas A&M University
 */
 
 
#include <iosfwd> 
#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// A muon DT SuperLayer ( an AlignableDet )

class AlignableDTSuperLayer: public AlignableDet {
 public:
  friend std::ostream& operator<< (std::ostream&, const AlignableDTSuperLayer &);

  /// Constructor
  AlignableDTSuperLayer(const GeomDet *geomDet);
};

#endif  // ALIGNABLE_DT_SUPERLAYER_H

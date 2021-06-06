#ifndef Alignment_MuonAlignment_AlignableGEMChamber_H
#define Alignment_MuonAlignment_AlignableGEMChamber_H

/* \class AlignableGEMChamber
 * \author Hyunyong Kim - TAMU
 */

#include <iosfwd>
#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

class AlignableGEMChamber : public AlignableDet {
public:
  friend std::ostream& operator<<(std::ostream&, const AlignableGEMChamber&);

  AlignableGEMChamber(const GeomDet* geomDet);

  void update(const GeomDet* geomDet);
};

#endif

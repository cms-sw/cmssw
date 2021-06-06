#ifndef Alignment_MuonAlignment_AlignableGEMSuperChamber_H
#define Alignment_MuonAlignment_AlignableGEMSuperChamber_H

/* \class AlignableGEMSuperChamber
 * \author Hyunyong Kim - TAMU
 */

#include <iosfwd>
#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

class AlignableGEMSuperChamber : public AlignableDet {
public:
  friend std::ostream& operator<<(std::ostream&, const AlignableGEMSuperChamber&);

  AlignableGEMSuperChamber(const GeomDet* geomDet);

  void update(const GeomDet* geomDet);
};

#endif

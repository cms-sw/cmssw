#ifndef Alignment_MuonAlignment_AlignableGEMStation_H
#define Alignment_MuonAlignment_AlignableGEMStation_H

/* \class AlignableGEMRing
 * \author Hyunyong Kim - TAMU
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableGEMRing.h"

#include <vector>

class GeomDet;
class AlignableGEMRing;

class AlignableGEMStation : public AlignableComposite {
public:
  AlignableGEMStation(const std::vector<AlignableGEMRing*>& GEMRings);

  PositionType computePosition();

  RotationType computeOrientation();

  AlignableSurface computeSurface();

  AlignableGEMRing& ring(int i);

  friend std::ostream& operator<<(std::ostream&, const AlignableGEMStation&);

  void dump(void) const override;

private:
  std::vector<AlignableGEMRing*> theGEMRings;
};

#endif

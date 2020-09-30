#ifndef Alignment_MuonAlignment_AlignableGEMRing_H
#define Alignment_MuonAlignment_AlignableGEMRing_H

/* \class AlignableGEMRing
 * \author Hyunyong Kim - TAMU
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"
#include "Alignment/MuonAlignment/interface/AlignableGEMSuperChamber.h"
#include <vector>

class GeomDet;

class AlignableGEMRing : public AlignableComposite {
public:
  AlignableGEMRing(const std::vector<AlignableGEMSuperChamber*>& GEMSuperChambers);

  PositionType computePosition();

  RotationType computeOrientation();

  AlignableSurface computeSurface();

  AlignableGEMSuperChamber& superChamber(int i);

  friend std::ostream& operator<<(std::ostream&, const AlignableGEMRing&);

  void dump(void) const override;

private:
  std::vector<AlignableGEMSuperChamber*> theGEMSuperChambers;
};

#endif

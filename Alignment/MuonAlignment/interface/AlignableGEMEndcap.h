#ifndef Alignment_MuonAlignment_AlignableGEMEndcap_H
#define Alignment_MuonAlignment_AlignableGEMEndcap_H

/* \class AlignableGEMEndcap
 * \author Hyunyong Kim - TAMU
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableGEMStation.h"

#include <vector>

class GeomDet;

class AlignableGEMEndcap : public AlignableComposite {
public:
  AlignableGEMEndcap(const std::vector<AlignableGEMStation*>& GEMStations);

  PositionType computePosition();

  RotationType computeOrientation();

  AlignableSurface computeSurface();

  AlignableGEMStation& station(int i);

  friend std::ostream& operator<<(std::ostream&, const AlignableGEMEndcap&);

  void dump(void) const override;

  Alignments* alignments() const override;

  AlignmentErrorsExtended* alignmentErrors() const override;

private:
  std::vector<AlignableGEMStation*> theGEMStations;
};

#endif

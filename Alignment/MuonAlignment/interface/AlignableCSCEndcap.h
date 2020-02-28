#ifndef Alignment_MuonAlignment_AlignableCSCEndcap_H
#define Alignment_MuonAlignment_AlignableCSCEndcap_H

/** \class AlignableCSCCEndcap
 *  The alignable muon CSC endcap.
 *
 *  $Date: 2008/04/15 16:05:53 $
 *  $Revision: 1.10 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"

#include <vector>

class GeomDet;

/// Concrete class for muon CSC Endcap alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableCSCEndcap : public AlignableComposite {
public:
  AlignableCSCEndcap(const std::vector<AlignableCSCStation*>& cscStations);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableCSCStation& station(int i);

  /// Printout muon End Cap information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableCSCEndcap&);

  /// Recursive printout of the muon End Cap structure
  void dump(void) const override;

  // Get alignments sorted by DetId
  Alignments* alignments() const override;

  // Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const override;

private:
  std::vector<AlignableCSCStation*> theCSCStations;
};

#endif

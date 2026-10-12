#ifndef Alignment_MTDAlignment_AlignableBTL_H
#define Alignment_MTDAlignment_AlignableBTL_H

/** \class AlignableBTL
 *  The alignable BTL 
 *
 *  $Date: 2024/12/15 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableBTLTray.h"

#include <vector>

class GeomDet;

/// Concrete class for BTL alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableBTL : public AlignableComposite {
public:
  AlignableBTL(const std::vector<AlignableBTLTray*>& btlTrays);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();

  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"

  // get the Surface
  AlignableSurface computeSurface();

  AlignableBTLTray& tray(int i);

  /// Printout btl information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableBTL&);

  /// Recursive printout of the btl structure
  void dump(void) const override;

  // Get alignments sorted by DetId
  Alignments* alignments() const override;

  // Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const override;

private:
  std::vector<AlignableBTLTray*> theBTLTrays;
};

#endif

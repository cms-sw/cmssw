#ifndef Alignment_MTDAlignment_AlignableETLDee_H
#define Alignment_MTDAlignment_AlignableETLDee_H

/** \class AlignableETLDee
 *
 *  $Date: 2024/10/27 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableETLServiceHybrid.h"

#include <vector>

class GeomDet;

///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableETLDee : public AlignableComposite {
public:
  AlignableETLDee(const std::vector<AlignableETLServiceHybrid*>& etlServiceHybrids);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableETLServiceHybrid& serviceHybrid(int i);

  /// Printout Dee information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableETLDee&);

  /// Recursive printout of the Dee structure
  void dump(void) const override;

  // Get alignments sorted by DetId
  Alignments* alignments() const override;

  // Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const override;

private:
  std::vector<AlignableETLServiceHybrid*> theETLServiceHybrids;
};

#endif

#ifndef Alignment_MTDAlignment_AlignableETLServiceHybrid_H
#define Alignment_MTDAlignment_AlignableETLServiceHybrid_H

/** \class AlignableETLServiceHybrid
 *
 *  $Date: 2024/10/27 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableETLModule.h"

#include <vector>

class GeomDet;

///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableETLServiceHybrid : public AlignableComposite {
public:
  AlignableETLServiceHybrid(const std::vector<AlignableETLModule*>& etlModules);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableETLModule& mod(int i);

  /// Printout Hybrid Service information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableETLServiceHybrid&);

  /// Recursive printout of the Service Hybrid structure
  void dump(void) const override;

  // Get alignments sorted by DetId
  Alignments* alignments() const override;

  // Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const override;

private:
  std::vector<AlignableETLModule*> theETLModules;
};

#endif

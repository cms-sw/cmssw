#ifndef Alignment_MTDAlignment_AlignableETLDisk_H
#define Alignment_MTDAlignment_AlignableETLDisk_H

/** \class AlignableETLDisk
 *
 *  $Date: 2024/10/27 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableETLDee.h"

#include <vector>

class GeomDet;

///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableETLDisk : public AlignableComposite {
public:
  AlignableETLDisk(const std::vector<AlignableETLDee*>& etlDees);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableETLDee& dee(int i);

  /// Printout Disk information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableETLDisk&);

  /// Recursive printout of the Disk structure
  void dump(void) const override;

private:
  std::vector<AlignableETLDee*> theETLDees;
};

#endif

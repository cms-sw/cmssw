#ifndef Alignment_MTDAlignment_AlignableETLEndcap_H
#define Alignment_MTDAlignment_AlignableETLEndcap_H

/** \class AlignableETLEndcap
 *
 *  $Date: 2024/10/27 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableETLDisk.h"

#include <vector>

class GeomDet;

///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableETLEndcap : public AlignableComposite {
public:
  AlignableETLEndcap(const std::vector<AlignableETLDisk*>& etlDisks);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableETLDisk& disk(int i);

  /// Printout Endcap information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableETLEndcap&);

  /// Recursive printout of the Endcap structure
  void dump(void) const override;

private:
  std::vector<AlignableETLDisk*> theETLDisks;
};

#endif

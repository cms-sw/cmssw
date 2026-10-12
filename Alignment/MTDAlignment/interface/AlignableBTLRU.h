#ifndef Alignment_MTDAlignment_AlignableBTLRU_H
#define Alignment_MTDAlignment_AlignableBTLRU_H

/** \class AlignableBTLRU
 *  The alignable BTL RU.
 *
 *  $Date: 2024/12/15 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableBTLModule.h"

#include <vector>

class GeomDet;

/// Concrete class for BTL RU alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableBTLRU : public AlignableComposite {
public:
  AlignableBTLRU(const std::vector<AlignableBTLModule*>& btlModules);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableBTLModule& mod(int i);

  /// Printout BTL RU information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableBTLRU&);

  /// Recursive printout of the BTL RU structure
  void dump(void) const override;

private:
  std::vector<AlignableBTLModule*> theBTLModules;
};

#endif

#ifndef Alignment_MTDAlignment_AlignableBTLTray_H
#define Alignment_MTDAlignment_AlignableBTLTray_H

/** \class AlignableBTLTray
 *  The alignable BTL tray.
 *
 *  $Date: 2024/10/27 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableBTLRU.h"

#include <vector>

class GeomDet;

/// Concrete class for BTL Tray alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableBTLTray : public AlignableComposite {
public:
  AlignableBTLTray(const std::vector<AlignableBTLRU*>& btlRUs);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableBTLRU& ru(int i);

  /// Printout Tray information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableBTLTray&);

  /// Recursive printout of the Tray structure
  void dump(void) const override;

private:
  std::vector<AlignableBTLRU*> theBTLRUs;
};

#endif

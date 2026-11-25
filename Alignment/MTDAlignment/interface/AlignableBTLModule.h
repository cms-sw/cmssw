#ifndef Alignment_MTDAlignment_AlignableBTLModule_H
#define Alignment_MTDAlignment_AlignableBTLModule_H

/** \class AlignableBTLModule
 *  The alignable BTL RU.
 *
 *  $Date: 2024/12/15 16:05:53 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MTDAlignment/interface/AlignableBTLSensorModule.h"

#include <vector>

class GeomDet;

/// Concrete class for BTL RU alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableBTLModule : public AlignableComposite {
public:
  AlignableBTLModule(const std::vector<AlignableBTLSensorModule*>& btlSensorModules);

  // gets the global position as the average over all positions of the layers
  PositionType computePosition();
  // get the global orientation
  RotationType computeOrientation();  //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  AlignableBTLSensorModule& sensormod(int i);

  /// Printout BTL Module information (not recursive)
  friend std::ostream& operator<<(std::ostream&, const AlignableBTLModule&);

  /// Recursive printout of the BTL RU structure
  void dump(void) const override;

private:
  std::vector<AlignableBTLSensorModule*> theBTLSensorModules;
};

#endif

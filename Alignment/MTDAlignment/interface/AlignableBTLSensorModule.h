#ifndef Alignment_MTDAlignment_AlignableBTLSensorModule_H
#define Alignment_MTDAlignment_AlignableBTLSensorModule_H

/** \class AlignableBTLSensorModule
 *  The alignable BTL module.
 *
 *  $Date: 2024/12/14 09:39:20 $
 *  $Revision: 1.0 $
 *  \author Pablo Martínez Ruiz del Arbol - IFCA
 */

#include <iosfwd>
#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// A BTL Module ( an AlignableDet )

class AlignableBTLSensorModule : public AlignableDet {
public:
  friend std::ostream &operator<<(std::ostream &, const AlignableBTLSensorModule &);

  /// Constructor
  AlignableBTLSensorModule(const GeomDet *geomDet);

  void dump(void) const override;
};

#endif  // ALIGNABLE_BTL_MODULE_H

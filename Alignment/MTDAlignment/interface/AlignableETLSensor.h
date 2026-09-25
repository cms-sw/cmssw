#ifndef Alignment_MTDAlignment_AlignableETLSensor_H
#define Alignment_MTDAlignment_AlignableETLSensor_H

/** \class AlignableETLSensor
 *  The alignable ETL module.
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// An ETL Sensor ( an AlignableDet )

class AlignableETLSensor : public AlignableDet {
public:
  friend std::ostream &operator<<(std::ostream &, const AlignableETLSensor &);

  void dump(void) const override;

  /// Constructor
  AlignableETLSensor(const GeomDet *geomDet);
};

#endif  // ALIGNABLE_ETL_MODULE_H

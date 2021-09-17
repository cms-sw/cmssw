///////////////////////////////////////////////////////////////////////////////
// File: EcalNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for ECal
///////////////////////////////////////////////////////////////////////////////

#ifndef EcalNumberingScheme_h
#define EcalNumberingScheme_h

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdint>

class EcalNumberingScheme : public CaloNumberingScheme {
public:
  EcalNumberingScheme();
  ~EcalNumberingScheme() override;
  virtual uint32_t getUnitID(const EcalBaseNumber& baseNumber) const = 0;
};

#endif

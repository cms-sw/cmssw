///////////////////////////////////////////////////////////////////////////////
// File: EcalShashlikNumberingScheme.h
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalShashlikNumberingScheme_h
#define EcalShashlikNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class EcalShashlikNumberingScheme : public EcalNumberingScheme {

public:
  EcalShashlikNumberingScheme();
  ~EcalShashlikNumberingScheme();
  virtual uint32_t getUnitID(const EcalBaseNumber& baseNumber) const ;

};

#endif

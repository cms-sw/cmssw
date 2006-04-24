///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.h
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalEndcapNumberingScheme_h
#define EcalEndcapNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class EcalEndcapNumberingScheme : public EcalNumberingScheme {

public:
  EcalEndcapNumberingScheme();
  ~EcalEndcapNumberingScheme();
  virtual uint32_t getUnitID(const EcalBaseNumber& baseNumber) const ;

};

#endif

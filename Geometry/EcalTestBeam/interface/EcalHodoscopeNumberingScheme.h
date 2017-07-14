///////////////////////////////////////////////////////////////////////////////
// File: EcalHodoscopeNumberingScheme.h
// Description: Numbering scheme for TB H4 hodoscope detector
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalHodoscopeNumberingScheme_h
#define EcalHodoscopeNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class EcalHodoscopeNumberingScheme : public EcalNumberingScheme {

 public:

  EcalHodoscopeNumberingScheme();
  ~EcalHodoscopeNumberingScheme() override;
  uint32_t getUnitID(const EcalBaseNumber& baseNumber) const override ;

private:

};

#endif

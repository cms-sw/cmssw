#ifndef ESTBNumberingScheme_h
#define ESTBNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class ESTBNumberingScheme : public EcalNumberingScheme {

public:

  ESTBNumberingScheme();
  ~ESTBNumberingScheme() override;
  uint32_t getUnitID(const EcalBaseNumber& baseNumber) const override ;

private:

  int iX[30];
  int iY[30];

};

#endif

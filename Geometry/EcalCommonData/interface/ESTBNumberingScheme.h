#ifndef ESTBNumberingScheme_h
#define ESTBNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class ESTBNumberingScheme : public EcalNumberingScheme {

public:

  ESTBNumberingScheme();
  ~ESTBNumberingScheme();
  virtual uint32_t getUnitID(const EcalBaseNumber& baseNumber) const ;

private:

  int iX[30];
  int iY[30];

};

#endif

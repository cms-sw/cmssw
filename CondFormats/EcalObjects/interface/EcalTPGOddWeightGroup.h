#ifndef EcalTPGOddWeightGroup_h
#define EcalTPGOddWeightGroup_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"

/*

P.P.
*/

class EcalTPGOddWeightGroup : public EcalTPGGroups {
public:
  EcalTPGOddWeightGroup();
  ~EcalTPGOddWeightGroup();

  COND_SERIALIZABLE;
};

#endif

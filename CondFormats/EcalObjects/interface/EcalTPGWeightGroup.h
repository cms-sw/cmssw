#ifndef EcalTPGWeightGroup_h
#define EcalTPGWeightGroup_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"

class EcalTPGWeightGroup : public EcalTPGGroups {
public:
  EcalTPGWeightGroup();
  ~EcalTPGWeightGroup();

  COND_SERIALIZABLE;
};

#endif

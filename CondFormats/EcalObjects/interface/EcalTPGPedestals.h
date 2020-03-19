#ifndef EcalTPGPedestals_h
#define EcalTPGPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalTPGPedestal {
  EcalTPGPedestal() : mean_x12(0), mean_x6(0), mean_x1(0) {}
  uint32_t mean_x12;
  uint32_t mean_x6;
  uint32_t mean_x1;

  COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalTPGPedestal> EcalTPGPedestalsMap;
typedef EcalTPGPedestalsMap::const_iterator EcalTPGPedestalsMapIterator;
typedef EcalTPGPedestalsMap EcalTPGPedestals;

#endif

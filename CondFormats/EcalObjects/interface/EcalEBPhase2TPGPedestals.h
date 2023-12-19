#ifndef CondFormats_EcalObjects_EcalEBPhase2TPGPedestals_h
#define CondFormats_EcalObjects_EcalEBPhase2TPGPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalEBPhase2TPGPedestal {
  EcalEBPhase2TPGPedestal() : mean_x10(0), mean_x1(0) {}
  uint32_t mean_x10;
  uint32_t mean_x1;

  COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalEBPhase2TPGPedestal> EcalEBPhase2TPGPedestalsMap;
typedef EcalEBPhase2TPGPedestalsMap::const_iterator EcalEBPhase2TPGPedestalsMapIterator;
typedef EcalEBPhase2TPGPedestalsMap EcalEBPhase2TPGPedestals;

#endif

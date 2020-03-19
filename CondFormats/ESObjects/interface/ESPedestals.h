#ifndef ESPedestals_h
#define ESPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"

struct ESPedestal {
  struct Zero {
    float z1;
    float z2;
  };

  static const Zero zero;

  float mean;
  float rms;

public:
  float getMean() const { return mean; }

  float getRms() const { return rms; }

  COND_SERIALIZABLE;
};

typedef ESCondObjectContainer<ESPedestal> ESPedestalsMap;
typedef ESPedestalsMap::const_iterator ESPedestalsMapIterator;
typedef ESPedestalsMap ESPedestals;

#endif

#ifndef EcalPedestals_h
#define EcalPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalPedestal {
  EcalPedestal() : mean_x12(0), rms_x12(0), mean_x6(0), rms_x6(0), mean_x1(0), rms_x1(0) {}

  struct Zero {
    Zero() : z1(0), z2(0) {}
    float z1;
    float z2;
  };

  static const Zero zero;

  float mean_x12;
  float rms_x12;
  float mean_x6;
  float rms_x6;
  float mean_x1;
  float rms_x1;

public:
  float const* mean_rms(int i) const {
    if (i == 0)
      return &zero.z1;
    return (&mean_x12) + (2 * (i - 1));
  }

  float mean(int i) const {
    if (i == 0)
      return 0.;
    return *(&mean_x12 + (2 * (i - 1)));
  }

  float rms(int i) const {
    if (i == 0)
      return 0.;
    return *(&rms_x12 + (2 * (i - 1)));
  }

  COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalPedestal> EcalPedestalsMap;
typedef EcalPedestalsMap::const_iterator EcalPedestalsMapIterator;
typedef EcalPedestalsMap EcalPedestals;

#endif

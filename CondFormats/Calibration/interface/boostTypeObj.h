#ifndef BOOSTTYPEOBJ_H
#define BOOSTTYPEOBJ_H
#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class boostTypeObj {
public:
  int8_t a;
  int16_t b;
  uint8_t aa;
  uint16_t bb;

  COND_SERIALIZABLE;
};
#endif

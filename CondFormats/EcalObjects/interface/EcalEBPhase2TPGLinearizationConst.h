#ifndef EcalEBPhase2TPGLinearizationConst_h
#define EcalEBPhase2TPGLinearizationConst_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalEBPhase2TPGLinearizationConstant {
  EcalEBPhase2TPGLinearizationConstant()
      : mult_x10(0), mult_x1(0), shift_x10(0), shift_x1(0), i2cSub_x10(0), i2cSub_x1(0) {}

  uint32_t mult_x10;
  uint32_t mult_x1;
  uint32_t shift_x10;
  uint32_t shift_x1;
  uint32_t i2cSub_x10;
  uint32_t i2cSub_x1;

  COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalEBPhase2TPGLinearizationConstant> EcalEBPhase2TPGLinearizationConstMap;
typedef EcalCondObjectContainer<EcalEBPhase2TPGLinearizationConstant>::const_iterator
    EcalEBPhase2TPGLinearizationConstMapIterator;
typedef EcalEBPhase2TPGLinearizationConstMap EcalEBPhase2TPGLinearizationConst;

#endif

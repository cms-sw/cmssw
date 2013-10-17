#ifndef EcalTPGLinearizationConst_h
#define EcalTPGLinearizationConst_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalTPGLinearizationConstant
{
   uint32_t mult_x12 ;
   uint32_t mult_x6 ;
   uint32_t mult_x1 ;
   uint32_t shift_x12 ;
   uint32_t shift_x6 ;
   uint32_t shift_x1 ;

   COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalTPGLinearizationConstant> EcalTPGLinearizationConstMap;
typedef EcalCondObjectContainer<EcalTPGLinearizationConstant>::const_iterator EcalTPGLinearizationConstMapIterator;
typedef EcalTPGLinearizationConstMap EcalTPGLinearizationConst;

#endif

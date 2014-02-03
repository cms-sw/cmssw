#ifndef EcalTPGPedestals_h
#define EcalTPGPedestals_h

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalTPGPedestal
{
   uint32_t mean_x12 ;
   uint32_t mean_x6 ;
   uint32_t mean_x1 ;
};

typedef EcalCondObjectContainer<EcalTPGPedestal> EcalTPGPedestalsMap;
typedef EcalTPGPedestalsMap::const_iterator EcalTPGPedestalsMapIterator;
typedef EcalTPGPedestalsMap EcalTPGPedestals;

#endif

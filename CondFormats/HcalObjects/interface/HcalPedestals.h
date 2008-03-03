#ifndef HcalPedestals_h
#define HcalPedestals_h

/** 
\class HcalPedestals
\author Radek Ofierzynski
POOL container to store Pedestal values 4xCapId, using template
*/

#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

typedef HcalCondObjectContainer<HcalPedestal> HcalPedestals;

#endif

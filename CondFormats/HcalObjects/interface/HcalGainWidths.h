#ifndef HcalGainWidths_h
#define HcalGainWidths_h

/** 
\class HcalGainWidths
\author Radek Ofierzynski
POOL container to store GainWidth values 4xCapId
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"

typedef HcalCondObjectContainer<HcalGainWidth> HcalGainWidths;

#endif

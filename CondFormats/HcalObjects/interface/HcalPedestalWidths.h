#ifndef HcalPedestalWidths_h
#define HcalPedestalWidths_h

/** 
\class HcalPedestalWidths
\author Radek Ofierzynski
POOL container to store PedestalWidth values 4xCapId, using template
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

typedef HcalCondObjectContainer<HcalPedestalWidth> HcalPedestalWidths;

#endif

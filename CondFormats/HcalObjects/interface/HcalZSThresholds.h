#ifndef HcalZSThresholds_h
#define HcalZSThresholds_h

/*
\class HcalZSThresholds
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalZSThreshold.h"

typedef HcalCondObjectContainer<HcalZSThreshold> HcalZSThresholds;

#endif

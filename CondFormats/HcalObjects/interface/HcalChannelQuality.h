#ifndef HcalChannelQuality_h
#define HcalChannelQuality_h

/** 
\class HcalChannelQuality
\author Radek Ofierzynski
POOL object to store HcalChannelStatus
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"

typedef HcalCondObjectContainer<HcalChannelStatus> HcalChannelQuality;


#endif


#ifndef CondFormats_EcalObjects_EcalDQMChannelStatus_H
#define CondFormats_EcalObjects_EcalDQMChannelStatus_H
/**
 **/

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalDQMStatusCode.h"

typedef EcalCondObjectContainer<EcalDQMStatusCode> EcalDQMChannelStatusMap;
typedef EcalDQMChannelStatusMap EcalDQMChannelStatus;

#endif

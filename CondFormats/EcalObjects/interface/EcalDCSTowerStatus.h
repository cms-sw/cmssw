#ifndef CondFormats_EcalObjects_EcalDCSTowerStatus_H
#define CondFormats_EcalObjects_EcalDCSTowerStatus_H
/**
 * Author: Francesca
 * Created: 15 December 2009
 * 
 **/

#include "CondFormats/EcalObjects/interface/EcalCondTowerObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

typedef EcalCondTowerObjectContainer<EcalChannelStatusCode> EcalDCSTowerStatusMap;
typedef EcalDCSTowerStatusMap EcalDCSTowerStatus;

#endif

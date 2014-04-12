#ifndef CondFormats_EcalObjects_EcalDQMTowerStatus_H
#define CondFormats_EcalObjects_EcalDQMTowerStatus_H
/**
 * Author: Francesca
 * Created: 15 December 2009
 * 
 **/

#include "CondFormats/EcalObjects/interface/EcalCondTowerObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalDQMStatusCode.h"

typedef EcalCondTowerObjectContainer<EcalDQMStatusCode> EcalDQMTowerStatusMap;
typedef EcalDQMTowerStatusMap EcalDQMTowerStatus;

#endif

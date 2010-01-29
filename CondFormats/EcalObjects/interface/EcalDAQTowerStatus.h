#ifndef CondFormats_EcalObjects_EcalDAQTowerStatus_H
#define CondFormats_EcalObjects_EcalDAQTowerStatus_H
/**
 * Author: Francesca
 * Created: 15 December 2009
 * 
 **/

#include "CondFormats/EcalObjects/interface/EcalCondTowerObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalDAQStatusCode.h"

typedef EcalCondTowerObjectContainer<EcalDAQStatusCode> EcalDAQTowerStatusMap;
typedef EcalDAQTowerStatusMap EcalDAQTowerStatus;

#endif

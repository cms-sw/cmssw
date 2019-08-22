#ifndef CondFormats_EcalObjects_EcalTPGCrystalStatus_H
#define CondFormats_EcalObjects_EcalTPGCrystalStatus_H
/**
 * Author: Francesca Cavallari
 * Created: 3 dec 2008
 * 
 **/

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatusCode.h"

typedef EcalCondObjectContainer<EcalTPGCrystalStatusCode> EcalTPGCrystalStatusMap;

typedef EcalTPGCrystalStatusMap::const_iterator EcalTPGCrystalStatusMapIterator;
typedef EcalTPGCrystalStatusMap EcalTPGCrystalStatus;

#endif

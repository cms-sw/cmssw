#ifndef EcalMappingElectronics_
#define EcalMappingElectronics_

/**
 * Author: Emmanuelle Perez & Paolo Meridiani
 * $Id: $
 **/
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalMappingElement {
  EcalMappingElement() : electronicsid(0), triggerid(0) {}
  uint32_t electronicsid;
  uint32_t triggerid;

  COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalMappingElement> EcalMappingElectronicsMap;
typedef EcalMappingElectronicsMap::const_iterator EcalMappingElectronicsMapIterator;
typedef EcalMappingElectronicsMap EcalMappingElectronics;

#endif

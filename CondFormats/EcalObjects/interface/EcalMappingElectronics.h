#ifndef EcalMappingElectronics_
#define EcalMappingElectronics_

/**
 * Author: Emmanuelle Perez & Paolo Meridiani
 * $Id: EcalMappingElectronics.h,v 1.1 2008/03/06 14:43:57 meridian Exp $
 **/
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalMappingElement
{
  uint32_t electronicsid;
  uint32_t triggerid;
};

typedef EcalCondObjectContainer<EcalMappingElement> EcalMappingElectronicsMap;
typedef EcalMappingElectronicsMap::const_iterator EcalMappingElectronicsMapIterator;
typedef EcalMappingElectronicsMap EcalMappingElectronics;

#endif


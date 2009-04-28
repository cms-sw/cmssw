//
// F.Ratnikov (UMd), Nov 1, 2005
//
#ifndef HcalZSXML_h
#define HcalZSXML_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

namespace HcalZSXML {
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
		   const HcalZSThresholds& fObject);
} 
#endif

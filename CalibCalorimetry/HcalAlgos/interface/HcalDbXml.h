//
// F.Ratnikov (UMd), Nov 1, 2005
//
#ifndef HcalDbXml_h
#define HcalDbXml_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllClasses.h"

/**
   \brief IO for XML instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbProducer.h,v 1.2 2005/10/04 18:03:03 fedor Exp $
   
*/
namespace HcalDbXml {
  bool dumpObject (std::ostream& fOutput, unsigned fRun, const std::string& fTag, const HcalPedestals& fObject, const HcalPedestalWidths& fError);
  bool dumpObject (std::ostream& fOutput, unsigned fRun, const std::string& fTag, const HcalGains& fObject, const HcalGainWidths& fError);
} 
#endif

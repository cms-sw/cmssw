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
   $Id: HcalDbXml.h,v 1.3 2006/01/21 01:32:19 fedor Exp $
   
*/
namespace HcalDbXml {
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
		   const HcalPedestals& fObject, const HcalPedestalWidths& fError);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
		   const HcalPedestals& fObject);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
		   const HcalGains& fObject, const HcalGainWidths& fError);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
		   const HcalGains& fObject);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
		   const HcalElectronicsMap& fObject) {return false;}
} 
#endif

//
// F.Ratnikov (UMd), Nov 1, 2005
//
#ifndef HcalDbXml_h
#define HcalDbXml_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

/**
   \brief IO for XML instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbXml.h,v 1.5 2008/03/03 20:33:18 rofierzy Exp $
   
*/
namespace HcalDbXml {
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalPedestals& fObject, const HcalPedestalWidths& fError);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag,
		   const HcalPedestals& fObject);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag,
		   const HcalPedestalWidths& fObject) {return false;}
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalGains& fObject, const HcalGainWidths& fError);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalGains& fObject);
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalRawGains& fObject);
  inline bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalGainWidths& fObject) {return false;}
  inline bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalElectronicsMap& fObject) {return false;}
  inline bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalQIEData& fObject) {return false;}
  inline bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		   const HcalCalibrationQIEData& fObject) {return false;}
} 
#endif

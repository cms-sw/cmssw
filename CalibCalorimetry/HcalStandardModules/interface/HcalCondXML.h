// S. Won, Northwestern University
// Replacement for HcalDbXml packages
//
#ifndef HcalCondXML_h
#define HcalCondXML_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

namespace HcalCondXML {
  //Pedestals and widths (always together!)
  bool dumpObject (std::ostream& fOutput, 
		   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
		   const HcalPedestals& fObject, const HcalPedestalWidths& fObject2);
  //ZSThresholds
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalZSThresholds& fObject);
  //RespCorrs
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalRespCorrs& fObject);

  //Gains
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalGains& fObject);
  //GainWidths
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalGainWidths& fObject);
  //QIEData
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalQIEData& fObject);
  //ChannelQuality
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalChannelQuality& fObject);
  //L1TriggerObjects
  bool dumpObject (std::ostream& fOutput,
                   unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                   const HcalL1TriggerObjects& fObject);

} 
#endif

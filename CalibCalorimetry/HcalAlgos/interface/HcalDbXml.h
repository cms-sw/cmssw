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
   
*/
namespace HcalDbXml {
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalPedestals& fObject,
                  const HcalPedestalWidths& fError);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalPedestals& fObject);
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalPedestalWidths& fObject) {
    return false;
  }
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalGains& fObject,
                  const HcalGainWidths& fError);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalGains& fObject);
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalGainWidths& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalElectronicsMap& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalQIEData& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalCalibrationQIEData& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalQIETypes& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const HcalFrontEndMap& fObject) {
    return false;
  }
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalSiPMParameters& fObject);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalSiPMCharacteristics& fObject);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalTPParameters& fObject);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const HcalTPChannelParameters& fObject);
}  // namespace HcalDbXml
#endif

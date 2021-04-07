#ifndef CastorDbXml_h
#define CastorDbXml_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "CondFormats/CastorObjects/interface/AllClasses.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

/**
   \brief IO for XML instances of Hcal/Castor Calibrations
*/
namespace CastorDbXml {
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const CastorPedestals& fObject,
                  const CastorPedestalWidths& fError);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const CastorPedestals& fObject);
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const CastorPedestalWidths& fObject) {
    return false;
  }
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const CastorGains& fObject,
                  const CastorGainWidths& fError);
  bool dumpObject(std::ostream& fOutput,
                  unsigned fRun,
                  unsigned long fGMTIOVBegin,
                  unsigned long fGMTIOVEnd,
                  const std::string& fTag,
                  unsigned fVersion,
                  const CastorGains& fObject);
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const CastorGainWidths& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const CastorElectronicsMap& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const CastorQIEData& fObject) {
    return false;
  }
  inline bool dumpObject(std::ostream& fOutput,
                         unsigned fRun,
                         unsigned long fGMTIOVBegin,
                         unsigned long fGMTIOVEnd,
                         const std::string& fTag,
                         unsigned fVersion,
                         const CastorCalibrationQIEData& fObject) {
    return false;
  }
}  // namespace CastorDbXml
#endif

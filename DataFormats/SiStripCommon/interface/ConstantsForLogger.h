// Last commit: $Id: ConstantsForLogger.h,v 1.4 2007/03/21 08:22:59 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForLogger_H
#define DataFormats_SiStripCommon_ConstantsForLogger_H

#include <string>

/** 
    @file ConstantsForLogger.h
    @brief Constants defining MessageLogger categories
*/

namespace sistrip { 
  
  static const std::string mlCabling_       = "SiStripCabling";
  static const std::string mlCommissioning_ = "SiStripCommissioning";
  static const std::string mlConfigDb_      = "SiStripConfigDb";
  static const std::string mlDigis_         = "SiStripDigis";
  static const std::string mlDqmCommon_     = "SiStripDqmCommon";
  static const std::string mlDqmClient_     = "SiStripDqmClient";
  static const std::string mlDqmSource_     = "SiStripDqmSource";
  static const std::string mlESSources_     = "SiStripESSources";
  static const std::string mlInputSource_   = "SiStripInputSource";
  static const std::string mlO2O_           = "SiStripO2O";
  static const std::string mlRawToCluster_  = "SiStripRawToCluster";
  static const std::string mlRawToDigi_     = "SiStripRawToDigi";
  static const std::string mlSummaryPlots_  = "SiStripSummaryPlots";
  static const std::string mlTest_          = "SiStripTEST";

}

#endif // DataFormats_SiStripCommon_ConstantsForLogger_H

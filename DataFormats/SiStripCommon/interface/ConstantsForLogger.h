// Last commit: $Id: ConstantsForLogger.h,v 1.5 2007/03/26 10:11:53 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForLogger_H
#define DataFormats_SiStripCommon_ConstantsForLogger_H

#include <string>

/** 
    @file ConstantsForLogger.h
    @brief Constants defining MessageLogger categories
*/

namespace sistrip { 
  
  static const char mlCabling_[]       = "SiStripCabling";
  static const char mlCommissioning_[] = "SiStripCommissioning";
  static const char mlConfigDb_[]      = "SiStripConfigDb";
  static const char mlDigis_[]         = "SiStripDigis";
  static const char mlDqmCommon_[]     = "SiStripDqmCommon";
  static const char mlDqmClient_[]     = "SiStripDqmClient";
  static const char mlDqmSource_[]     = "SiStripDqmSource";
  static const char mlESSources_[]     = "SiStripESSources";
  static const char mlInputSource_[]   = "SiStripInputSource";
  static const char mlO2O_[]           = "SiStripO2O";
  static const char mlRawToCluster_[]  = "SiStripRawToCluster";
  static const char mlRawToDigi_[]     = "SiStripRawToDigi";
  static const char mlSummaryPlots_[]  = "SiStripSummaryPlots";
  static const char mlTest_[]          = "SiStripTEST";

}

#endif // DataFormats_SiStripCommon_ConstantsForLogger_H

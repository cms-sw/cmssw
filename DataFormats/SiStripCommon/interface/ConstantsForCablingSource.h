// Last commit: $Id: $

#ifndef DataFormats_SiStripCommon_ConstantsForCablingSource_H
#define DataFormats_SiStripCommon_ConstantsForCablingSource_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForCablingSource.h

    @brief Constants and enumerated type for defining the various
    "sources" of cabling object information.
*/
namespace sistrip { 

  // ---------- Constants ---------- 

  static const std::string unknownCablingSource_   = "UnknownCablingSource";
  static const std::string undefinedCablingSource_ = "UndefinedCablingSource";
  
  static const std::string cablingFromConns_   = "CablingFromConnections";
  static const std::string cablingFromDevices_ = "CablingFromDevices";
  static const std::string cablingFromDetIds_  = "CablingFromDetIds";

  // ---------- Enumerated type ---------- 
  
  enum CablingSource { UNKNOWN_CABLING_SOURCE   = sistrip::unknown_,
		       UNDEFINED_CABLING_SOURCE = sistrip::invalid_,
		       CABLING_FROM_CONNS       = 1,
		       CABLING_FROM_DEVICES     = 2,
		       CABLING_FROM_DETIDS      = 3
  };
  
}

#endif // DataFormats_SiStripCommon_ConstantsForCablingSource_H


// Last commit: $Id: ConstantsForCablingSource.h,v 1.3 2009/02/10 21:45:54 lowette Exp $

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

  static const char unknownCablingSource_[]   = "UnknownCablingSource";
  static const char undefinedCablingSource_[] = "UndefinedCablingSource";
  
  static const char cablingFromConns_[]   = "CablingFromConnections";
  static const char cablingFromDevices_[] = "CablingFromDevices";
  static const char cablingFromDetIds_[]  = "CablingFromDetIds";

  // ---------- Enumerated type ---------- 
  
  enum CablingSource { UNKNOWN_CABLING_SOURCE   = sistrip::unknown_,
		       UNDEFINED_CABLING_SOURCE = sistrip::invalid_,
		       CABLING_FROM_CONNS       = 1,
		       CABLING_FROM_DEVICES     = 2,
		       CABLING_FROM_DETIDS      = 3
  };
  
}

#endif // DataFormats_SiStripCommon_ConstantsForCablingSource_H


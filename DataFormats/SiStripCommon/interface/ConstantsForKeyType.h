// Last commit: $Id: ConstantsForKeyType.h,v 1.3 2009/02/10 21:45:54 lowette Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForKeyType_H
#define DataFormats_SiStripCommon_ConstantsForKeyType_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForKeyType.h 

    @brief Constants and enumerated type for sistrip::KeyType
*/

namespace sistrip { 
  
  // ---------- Constants ---------- 

  static const char unknownKey_[]   = "UnknownKey";
  static const char undefinedKey_[] = "UndefinedKey";

  static const char fedKey_[] = "FedKey";
  static const char fecKey_[] = "FecKey";
  static const char detKey_[] = "DetKey";

  // ---------- Enumerated type ---------- 

  enum KeyType { UNKNOWN_KEY   = sistrip::unknown_,  
		 UNDEFINED_KEY = sistrip::invalid_,  
		 FED_KEY       = 1, 
		 FEC_KEY       = 2, 
		 DET_KEY       = 3 
  };
  
}

#endif // DataFormats_SiStripCommon_ConstantsForKeyType_H



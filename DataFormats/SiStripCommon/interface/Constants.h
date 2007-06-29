// Last commit: $Id: Constants.h,v 1.4 2007/06/19 12:16:52 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_Constants_H
#define DataFormats_SiStripCommon_Constants_H

#include "boost/cstdint.hpp"
#include <string>

/** 
    @file Constants.h
    @brief Generic constants
*/    

namespace sistrip { 
  
  static const uint32_t invalid32_ = 0xFFFFFFFF; 
  static const uint16_t invalid_   = 0xFFFF; // 65535
  static const uint16_t valid_     = 0xFFFD; // 65533

  static const uint16_t unknown_   = 0xFFFE; // 65534
  static const uint16_t maximum_   = 0x03FF; // 1023

  static const std::string null_   = "null";
  
}

#endif // DataFormats_SiStripCommon_Constants_H

// Last commit: $Id: $

#ifndef DataFormats_SiStripCommon_ConstantsForGranularity_H
#define DataFormats_SiStripCommon_ConstantsForGranularity_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForGranularity.h

    @brief Constants and enumerated type for sistrip::Granularity
*/

namespace sistrip { 
  
  // ---------- Constants ---------- 
  
  // misc granularity
  static const std::string unknownGranularity_   = "UnknownGranularity";
  static const std::string undefinedGranularity_ = "UndefinedGranularity";
  
  // system granularity
  static const std::string tracker_   = "Tracker";
  static const std::string partition_ = "Partition";
  static const std::string tib_       = "Tib";
  static const std::string tob_       = "Tob";
  static const std::string tec_       = "Tec";

  // sub-structure granularity
  static const std::string layer_  = "Layer";
  static const std::string rod_    = "Rod";
  static const std::string string_ = "String";
  static const std::string disk_   = "Disk";
  static const std::string petal_  = "Petal";
  static const std::string ring_   = "Ring";

  // module granularity  
  static const std::string module_  = "Module";
  static const std::string lldChan_ = "LldChannel";
  static const std::string apv_     = "Apv";

  // readout granularity
  static const std::string fedSystem_  = "FedSystem";
  static const std::string fedId_      = "FedId";
  static const std::string feUnit_     = "FrontEndUnit";
  static const std::string feChan_     = "FrontEndChannel";
  static const std::string fedApv_     = "FedApv";
  static const std::string fedChannel_ = "FedChannel";

  // control granularity
  static const std::string fecSystem_ = "FecSystem";
  static const std::string fecCrate_  = "FecCrate";
  static const std::string fecSlot_   = "FecSlot";
  static const std::string fecRing_   = "FecRing";
  static const std::string ccuAddr_   = "CcuAddr";
  static const std::string ccuChan_   = "CcuChan";
 
  // ---------- Enumerated type ---------- 

  enum Granularity { UNDEFINED_GRAN = sistrip::invalid_, 
		     UNKNOWN_GRAN   = sistrip::unknown_, 
		     
		     TRACKER   = 1, 
		     PARTITION = 2, 
		     TIB       = 3, 
		     TOB       = 4, 
		     TEC       = 5,
    
		     LAYER  =  6, 
		     ROD    =  7, 
		     STRING =  8, 
		     DISK   =  9, 
		     PETAL  = 10, 
		     RING   = 11,

		     MODULE   = 12, 
		     LLD_CHAN = 13, 
		     APV      = 14,

		     FED_SYSTEM  = 15, 
		     FED         = 16, 
		     FE_UNIT     = 17, 
		     FE_CHAN     = 18,
		     FED_APV     = 19,
		     FED_CHANNEL = 20,

		     FEC_SYSTEM = 21,
		     FEC_CRATE  = 22,
		     FEC_SLOT   = 23,
		     FEC_RING   = 24,
		     CCU_ADDR   = 25,
		     CCU_CHAN   = 26

  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForGranularity_H



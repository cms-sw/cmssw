// Last commit: $Id: ConstantsForGranularity.h,v 1.5 2009/02/10 21:45:54 lowette Exp $

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
  static const char unknownGranularity_[]   = "UnknownGranularity";
  static const char undefinedGranularity_[] = "UndefinedGranularity";
  
  // system granularity
  static const char tracker_[]   = "Tracker";
  static const char partition_[] = "Partition";
  static const char tib_[]       = "Tib";
  static const char tob_[]       = "Tob";
  static const char tec_[]       = "Tec";

  // sub-structure granularity
  static const char layer_[]  = "Layer";
  static const char rod_[]    = "Rod";
  static const char string_[] = "String";
  static const char disk_[]   = "Disk";
  static const char petal_[]  = "Petal";
  static const char ring_[]   = "Ring";

  // module granularity  
  static const char module_[]  = "Module";
  static const char lldChan_[] = "LldChannel";
  static const char apv_[]     = "Apv";

  // readout granularity
  static const char fedSystem_[]  = "FedSystem";
  static const char feDriver_[]   = "FrontEndDriver";
  static const char feUnit_[]     = "FrontEndUnit";
  static const char feChan_[]     = "FrontEndChannel";
  static const char fedApv_[]     = "FedApv";
  static const char fedChannel_[] = "FedChannel";

  // control granularity
  static const char fecSystem_[] = "FecSystem";
  static const char fecCrate_[]  = "FecCrate";
  static const char fecSlot_[]   = "FecSlot";
  static const char fecRing_[]   = "FecRing";
  static const char ccuAddr_[]   = "CcuAddr";
  static const char ccuChan_[]   = "CcuChan";
 
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
		     FE_DRIVER   = 16, 
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



#ifndef CalibCalorimetry_HcalTPGAlgos_HcalChannelQualityManager_h
#define CalibCalorimetry_HcalTPGAlgos_HcalChannelQualityManager_h
// -*- C++ -*-
//
// Package:     HcalTPGAlgos
// Class  :     HcalChannelQualityManager
// 
/**\class HcalChannelQualityManager HcalChannelQualityManager.h CalibCalorimetry/HcalTPGAlgos/interface/HcalChannelQualityManager.h

 Description: Provides info regarding HCAL channel quality

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Fri Jul 10 09:38:37 CEST 2009
// $Id$
//

#include "DataFormats/HcalDetId/interface/HcalDetId.h"


class HcalChannelQualityManager
{

   public:
      HcalChannelQualityManager();
      virtual ~HcalChannelQualityManager();

      bool isChannelMasked(DetId channel, bool testmode = false);

   private:
      HcalChannelQualityManager(const HcalChannelQualityManager&); // stop default

      const HcalChannelQualityManager& operator=(const HcalChannelQualityManager&); // stop default
};


#endif

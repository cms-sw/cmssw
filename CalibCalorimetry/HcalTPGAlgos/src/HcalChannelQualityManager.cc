// -*- C++ -*-
//
// Package:     HcalTPGAlgos
// Class  :     HcalChannelQualityManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Fri Jul 10 10:39:06 CEST 2009
// $Id$
//

#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalChannelQualityManager.h"

HcalChannelQualityManager::HcalChannelQualityManager()
{
}

// HcalChannelQualityManager::HcalChannelQualityManager(const HcalChannelQualityManager& rhs)
// {
//    // do actual copying here;
// }

HcalChannelQualityManager::~HcalChannelQualityManager()
{
}

bool HcalChannelQualityManager::isChannelMasked(DetId channel, bool testmode){
  bool isMasked = false;
  //
  //_____ true is returned for a few channels for testing
  //
  if (testmode){
    //detid subdet ieta iphi depth
    //13401380 HB -1 1 1
    // trigger: ieta iphi
    //13408717 -32 1
    if (channel.rawId() == 13401380 || channel.rawId() == 13408717){
      isMasked = true;
    }
  }
  //
  //_____ normal mode of operation: true for every channel masked in ChannelQuality
  //
  else{
  }
  return isMasked;
}

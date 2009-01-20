#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalZSThresholds.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h"

namespace {
  struct dictionary {
    HcalPedestals mypeds;
    std::vector<HcalPedestal> mypedsVec;
 
    HcalPedestalWidths mywidths;
    std::vector<HcalPedestalWidth> mywidthsVec;
 
    HcalGains mygains;
    std::vector<HcalGain> mygainsVec;
 
    HcalGainWidths mygwidths;
    std::vector<HcalGainWidth> mygwidthsVec;
 
    HcalQIEData myqie;
    std::vector<HcalQIECoder> myqievec;
 
    HcalCalibrationQIEData mycalqie;
    std::vector<HcalCalibrationQIECoder> mycalqieVec;
 
    HcalElectronicsMap mymap;
    std::vector<HcalElectronicsMap::PrecisionItem> mymap2;
    std::vector<HcalElectronicsMap::TriggerItem> mymap3;
 
    HcalChannelQuality myquality;
    std::vector<HcalChannelStatus> myqualityVec;
 
    HcalZSThresholds myth;
    std::vector<HcalZSThreshold> mythvec;
 
    HcalRespCorrs mycorrs;
    std::vector<HcalRespCorr> mycorrsVec;
 
    HcalL1TriggerObjects myL1trigs;
    std::vector<HcalL1TriggerObject> myL1trigsVec;
  };
}


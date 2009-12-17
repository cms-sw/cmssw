#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"

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
 
    HcalLUTCorrs mylutcorrs;
    std::vector<HcalLUTCorr> mylutcorrsVec;
 
    HcalPFCorrs mypfcorrs;
    std::vector<HcalPFCorr> mypfcorrsVec;
 
    HcalL1TriggerObjects myL1trigs;
    std::vector<HcalL1TriggerObject> myL1trigsVec;

    HcalTimeCorrs mytcorrs;
    std::vector<HcalTimeCorr> mytcorrsVec;

    HcalValidationCorrs myVcorrs;
    std::vector<HcalValidationCorr> myVcorrsVec;
  };
}


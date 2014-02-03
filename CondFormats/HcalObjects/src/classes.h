#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"

namespace {
  struct dictionary {

    HcalPedestals mypeds();
    std::vector<HcalPedestal> mypedsVec;
 
    HcalPedestalWidths mywidths();
    std::vector<HcalPedestalWidth> mywidthsVec;
 
    HcalGains mygains();
    std::vector<HcalGain> mygainsVec;
 
    HcalGainWidths mygwidths();
    std::vector<HcalGainWidth> mygwidthsVec;
 
    HcalQIEData myqie();
    std::vector<HcalQIECoder> myqievec;
 
    HcalCalibrationQIEData mycalqie();
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

    HcalLutMetadata myLutMetadata;
    std::vector<HcalLutMetadatum> myLutMetadatumVec;
    HcalLutMetadata::NonChannelData myLutNonChannelMetadata;

    HcalCholeskyMatrices myCholeskys;
    std::vector<HcalCholeskyMatrix> myCholeskysVec;

    HcalCovarianceMatrices myCovariances;
    std::vector<HcalCovarianceMatrix> myCovariancesVec;

    HcalDcsValues myDcsValues;
    std::vector<HcalDcsValue> myDcsValueVec;

    HcalDcsMap myDcsMap;
    std::vector<HcalDcsMap::Item> myDcsMapVec;
    
    HcalLongRecoParams myLongRecoParams;
    std::vector<HcalLongRecoParam> myLongRecoParamVec;
    std::vector<uint32_t> myUintVec;
    
    HcalRecoParams myRecoParams;
    std::vector<HcalRecoParam> myRecoParamVec;
    
    HcalMCParams myMCParams;
    std::vector<HcalMCParam> myMCParamsVec;

    // HF noise DB objects
    HcalFlagHFDigiTimeParams myHcalFlagHFDigiTimeParams;
    std::vector<HcalFlagHFDigiTimeParam> myHcalFlagHFDigiTimeParamVec;

    HcalTimingParams myTimingParams;
    std::vector<HcalTimingParam> myTimingParamVec;
  };
}


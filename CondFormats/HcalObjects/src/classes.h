#include "CondFormats/HcalObjects/src/headers.h"

namespace CondFormats_HcalObjects {
  struct dictionary {
    HcalZDCLowGainFractions myfracs();
    std::vector<HcalZDCLowGainFraction> myfracsVec;

    HcalPedestals mypeds();
    std::vector<HcalPedestal> mypedsVec;

    HcalPedestalWidths mywidths();
    std::vector<HcalPedestalWidth> mywidthsVec;

    HcalGains mygains();
    std::vector<HcalGain> mygainsVec;

    HcalGainWidths mygwidths();
    std::vector<HcalGainWidth> mygwidthsVec;

    HcalPFCuts mypfcuts();
    std::vector<HcalPFCut> mypfcutVec;

    HcalQIEData myqie();
    std::vector<HcalQIECoder> myqievec;

    HcalCalibrationQIEData mycalqie();
    std::vector<HcalCalibrationQIECoder> mycalqieVec;

    HcalQIETypes myqietype();
    std::vector<HcalQIEType> myqietypevec;

    HcalSiPMParameters mySiPMParameter();
    std::vector<HcalSiPMParameter> mySiPMParametervec;

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

    HcalFrontEndMap myfmap1;
    std::vector<HcalFrontEndMap::PrecisionItem> myfmap2;

    HcalSiPMCharacteristics mySiPMCharacteristics;
    std::vector<HcalSiPMCharacteristics::PrecisionItem> mySiPMCharacteristicvec;

    HcalTPParameters myTPParameters;

    HcalTPChannelParameters myTPChannelParameters();
    std::vector<HcalTPChannelParameter> myTPChannelParametervec;

    // OOT pileup correction objects
    std::map<std::string, AbsOOTPileupCorrection*> myInnerMap;
    std::map<std::string, std::map<std::string, AbsOOTPileupCorrection*> > myOuterMap;
    ScalingExponential myScalingExponential;
    PiecewiseScalingPolynomial myPiecewiseScalingPolynomial;
    OOTPileupCorrDataFcn myOOTPileupCorrDataFcn;
    OOTPileupCorrData myOOTPileupCorrData;
    DummyOOTPileupCorrection myDummyOOTPileupCorrection;
    OOTPileupCorrectionMapColl myOOTPileupCorrectionMapColl;
    OOTPileupCorrectionBuffer myOOTPileupCorrectionBuffer;

    // QIE8 input pulse representation objects
    HcalInterpolatedPulse myHcalInterpolatedPulse;
    std::vector<HcalInterpolatedPulse> myHcalInterpolatedPulseVec;
    HBHEChannelGroups myHBHEChannelGroups;
    HcalInterpolatedPulseColl myHcalInterpolatedPulseColl;

    // HBHE negative energy filter
    std::vector<PiecewiseScalingPolynomial> myPiecewiseScalingPolynomialVec;
    HBHENegativeEFilter myHBHENegativeEFilter;

    // Phase 1 HF algorithm configuration data
    HFPhase1PMTParams myHFPhase1PMTParams;
  };
}  // namespace CondFormats_HcalObjects

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"

#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorGains.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "CondFormats/CastorObjects/interface/CastorCalibrationQIEData.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParam.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorr.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"

namespace {
  struct dictionary {
    CastorPedestals mypeds;
    std::vector<CastorPedestal> mypedsVec;
 
    CastorPedestalWidths mywidths;
    std::vector<CastorPedestalWidth> mywidthsVec;
 
    CastorGains mygains;
    std::vector<CastorGain> mygainsVec;
 
    CastorGainWidths mygwidths;
    std::vector<CastorGainWidth> mygwidthsVec;
 
    CastorQIEData myqie;
    std::vector<CastorQIECoder> myqievec;
 
    CastorCalibrationQIEData mycalqie;
    std::vector<CastorCalibrationQIECoder> mycalqieVec;
 
    CastorElectronicsMap mymap;
    std::vector<CastorElectronicsMap::PrecisionItem> mymap2;
    std::vector<CastorElectronicsMap::TriggerItem> mymap3;
 
    CastorChannelQuality myquality;
    std::vector<CastorChannelStatus> myqualityVec;

    CastorRecoParam myrecoparam;
    std::vector<CastorRecoParam> myrecoparamVec;
    CastorRecoParams myrecoparams;
    
    CastorSaturationCorr mysatcorr;
    std::vector<CastorSaturationCorr> mysatcorrVec;
    CastorSaturationCorrs mysatcorrs;
  };
}


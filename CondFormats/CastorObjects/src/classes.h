#include "CondFormats/CastorObjects/src/headers.h"

namespace CondFormats_CastorObjects {
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
}  // namespace CondFormats_CastorObjects

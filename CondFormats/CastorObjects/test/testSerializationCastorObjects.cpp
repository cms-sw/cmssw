#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/CastorObjects/src/headers.h"

int main() {
  testSerialization<CastorCalibrationQIECoder>();
  testSerialization<CastorCalibrationQIEData>();
  testSerialization<CastorChannelQuality>();
  testSerialization<CastorChannelStatus>();
  //testSerialization<CastorCondObjectContainer>(); missing arguments
  testSerialization<CastorElectronicsMap>();
  testSerialization<CastorElectronicsMap::PrecisionItem>();
  testSerialization<CastorElectronicsMap::TriggerItem>();
  testSerialization<CastorGain>();
  testSerialization<CastorGainWidth>();
  testSerialization<CastorGainWidths>();
  testSerialization<CastorGains>();
  testSerialization<CastorPedestal>();
  testSerialization<CastorPedestalWidth>();
  testSerialization<CastorPedestalWidths>();
  testSerialization<CastorPedestals>();
  testSerialization<CastorQIECoder>();
  testSerialization<CastorQIEData>();
  testSerialization<CastorRecoParam>();
  testSerialization<CastorRecoParams>();
  testSerialization<CastorSaturationCorr>();
  testSerialization<CastorSaturationCorrs>();
  testSerialization<std::vector<CastorCalibrationQIECoder>>();
  testSerialization<std::vector<CastorChannelStatus>>();
  testSerialization<std::vector<CastorElectronicsMap::PrecisionItem>>();
  testSerialization<std::vector<CastorElectronicsMap::TriggerItem>>();
  testSerialization<std::vector<CastorGain>>();
  testSerialization<std::vector<CastorGainWidth>>();
  testSerialization<std::vector<CastorPedestal>>();
  testSerialization<std::vector<CastorPedestalWidth>>();
  testSerialization<std::vector<CastorQIECoder>>();
  testSerialization<std::vector<CastorRecoParam>>();
  testSerialization<std::vector<CastorSaturationCorr>>();

  return 0;
}

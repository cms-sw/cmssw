#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/ESObjects/src/headers.h"

int main() {
  testSerialization<ESADCToGeVConstant>();
  testSerialization<ESAngleCorrectionFactors>();
  testSerialization<ESChannelStatus>();
  testSerialization<ESChannelStatusCode>();
  testSerialization<ESEEIntercalibConstants>();
  testSerialization<ESGain>();
  testSerialization<ESIntercalibConstants>();
  testSerialization<ESMIPToGeVConstant>();
  testSerialization<ESMissingEnergyCalibration>();
  testSerialization<ESPedestal>();
  testSerialization<ESPedestals>();
  testSerialization<ESRecHitRatioCuts>();
  testSerialization<ESStripGroupId>();
  testSerialization<ESTBWeights>();
  testSerialization<ESThresholds>();
  testSerialization<ESTimeSampleWeights>();
  testSerialization<ESWeightSet>();
  testSerialization<ESWeightStripGroups>();
  testSerialization<EcalContainer<ESDetId, ESChannelStatusCode>>();
  testSerialization<EcalContainer<ESDetId, ESPedestal>>();
  testSerialization<EcalContainer<ESDetId, ESStripGroupId>>();
  testSerialization<EcalContainer<ESDetId, float>>();
  testSerialization<std::map<ESStripGroupId, ESWeightSet>>();
  testSerialization<std::pair<ESStripGroupId, ESWeightSet>>();
  testSerialization<std::vector<ESChannelStatusCode>>();
  testSerialization<std::vector<ESPedestal>>();
  testSerialization<std::vector<ESStripGroupId>>();

  return 0;
}

#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/RunInfo/src/headers.h"

int main() {
  testSerialization<FillInfo>();
  testSerialization<LHCInfo>();
  testSerialization<LHCInfoPerLS>();
  testSerialization<LHCInfoPerFill>();
  testSerialization<LHCInfoVectorizedFields>();
  testSerialization<L1TriggerScaler>();
  testSerialization<L1TriggerScaler::Lumi>();
  testSerialization<MixingInputConfig>();
  testSerialization<MixingModuleConfig>();
  testSerialization<RunInfo>();
  testSerialization<RunSummary>();
  testSerialization<runinfo_test::RunNumber>();
  testSerialization<runinfo_test::RunNumber::Item>();
  testSerialization<std::bitset<3565>>();
  testSerialization<std::vector<L1TriggerScaler::Lumi>>();
  testSerialization<std::vector<MixingInputConfig>>();
  testSerialization<std::vector<runinfo_test::RunNumber::Item>>();

  return 0;
}

#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/HLTObjects/src/headers.h"

int main() {
  testSerialization<AlCaRecoTriggerBits>();
  testSerialization<std::pair<const std::string, std::vector<unsigned int>>>();
  //testSerialization<trigger::HLTPrescaleTableCond>(); never serialized in the old DB
  testSerialization<L1TObjScalingConstants>();
  testSerialization<std::vector<L1TObjScalingConstants>>();
  testSerialization<std::vector<L1TObjScalingConstants::Scaling>>();
  return 0;
}

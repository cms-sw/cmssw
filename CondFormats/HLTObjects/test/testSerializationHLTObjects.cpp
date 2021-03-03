#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/HLTObjects/src/headers.h"

int main() {
  testSerialization<AlCaRecoTriggerBits>();
  testSerialization<std::pair<const std::string, std::vector<unsigned int>>>();
  //testSerialization<trigger::HLTPrescaleTableCond>(); never serialized in the old DB

  return 0;
}

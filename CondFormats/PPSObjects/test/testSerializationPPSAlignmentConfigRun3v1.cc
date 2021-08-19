#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main() {
  testSerialization<PPSAlignmentConfigRun3v1::PointErrors>();
  testSerialization<PPSAlignmentConfigRun3v1::SelectionRange>();
  testSerialization<PPSAlignmentConfigRun3v1::RPConfig>();
  testSerialization<PPSAlignmentConfigRun3v1::SectorConfig>();
  testSerialization<PPSAlignmentConfigRun3v1::Binning>();

  testSerialization<std::vector<PPSAlignmentConfigRun3v1::PointErrors>>();
  testSerialization<std::map<unsigned int, std::vector<PPSAlignmentConfigRun3v1::PointErrors>>>();
  testSerialization<std::map<unsigned int, PPSAlignmentConfigRun3v1::SelectionRange>>();

  testSerialization<PPSAlignmentConfigRun3v1>();
}

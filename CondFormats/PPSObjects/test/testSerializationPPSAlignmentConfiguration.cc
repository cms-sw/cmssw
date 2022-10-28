#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main() {
  testSerialization<PPSAlignmentConfiguration::PointErrors>();
  testSerialization<PPSAlignmentConfiguration::SelectionRange>();
  testSerialization<PPSAlignmentConfiguration::RPConfig>();
  testSerialization<PPSAlignmentConfiguration::SectorConfig>();
  testSerialization<PPSAlignmentConfiguration::Binning>();

  testSerialization<std::vector<PPSAlignmentConfiguration::PointErrors>>();
  testSerialization<std::map<unsigned int, std::vector<PPSAlignmentConfiguration::PointErrors>>>();
  testSerialization<std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>>();

  testSerialization<PPSAlignmentConfiguration>();
}

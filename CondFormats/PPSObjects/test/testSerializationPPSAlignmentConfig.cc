#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main() {
  testSerialization<PointErrors>();
  testSerialization<SelectionRange>();
  testSerialization<RPConfig>();
  testSerialization<SectorConfig>();
  testSerialization<Binning>();

  testSerialization<std::vector<PointErrors>>();
  testSerialization<std::map<unsigned int, std::vector<PointErrors>>>();
  testSerialization<std::map<unsigned int, SelectionRange>>();

  testSerialization<PPSAlignmentConfig>();
}

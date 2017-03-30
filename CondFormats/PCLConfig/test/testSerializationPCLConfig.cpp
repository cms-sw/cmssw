#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
  testSerialization<AlignPCLThresholds>();
  testSerialization<std::vector<AlignPCLThreshold>>();
  //testSerialization<std::vector<AlignPCLThreshold::coordThresholds>>();
  return 0;
}

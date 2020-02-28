#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSet.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSetCollection.h"

int main() {
  testSerialization<LHCOpticalFunctionsSet>();
  testSerialization<LHCOpticalFunctionsSetCollection>();

  return 0;
}

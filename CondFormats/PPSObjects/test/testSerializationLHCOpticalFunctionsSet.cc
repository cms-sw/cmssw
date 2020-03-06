#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSet.h"
#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSetCollection.h"

int main() {
  testSerialization<LHCOpticalFunctionsSet>();
  testSerialization<LHCOpticalFunctionsSetCollection>();

  return 0;
}

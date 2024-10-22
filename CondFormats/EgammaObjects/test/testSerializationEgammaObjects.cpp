#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/EgammaObjects/src/headers.h"

int main() {
  testSerialization<ElectronLikelihoodCalibration>();
  testSerialization<ElectronLikelihoodCalibration::Entry>();
  testSerialization<ElectronLikelihoodCategoryData>();
  testSerialization<std::vector<ElectronLikelihoodCalibration::Entry>>();

  return 0;
}

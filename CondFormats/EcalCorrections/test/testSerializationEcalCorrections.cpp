#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/EcalCorrections/src/headers.h"

int main() {
  testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta>();
  testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients>();
  testSerialization<EcalShowerContainmentCorrections>();
  testSerialization<EcalShowerContainmentCorrections::Coefficients>();

  return 0;
}

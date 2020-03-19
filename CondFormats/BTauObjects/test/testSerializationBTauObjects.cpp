#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/BTauObjects/src/headers.h"

int main() {
  testSerialization<CombinedSVCalibration>();
  testSerialization<CombinedSVCalibration::Entry>();
  testSerialization<CombinedSVCategoryData>();
  testSerialization<CombinedTauTagCalibration>();
  testSerialization<CombinedTauTagCalibration::Entry>();
  testSerialization<CombinedTauTagCategoryData>();
  testSerialization<TrackProbabilityCalibration>();
  testSerialization<TrackProbabilityCalibration::Entry>();
  testSerialization<TrackProbabilityCategoryData>();
  testSerialization<std::vector<CombinedSVCalibration::Entry>>();
  testSerialization<std::vector<CombinedTauTagCalibration::Entry>>();
  testSerialization<std::vector<TrackProbabilityCalibration::Entry>>();

  testSerialization<BTagEntry>();
  testSerialization<BTagEntry::Parameters>();
  testSerialization<BTagCalibration>();

  return 0;
}

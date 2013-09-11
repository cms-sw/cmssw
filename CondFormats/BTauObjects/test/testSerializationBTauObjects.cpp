#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/BTauObjects/interface/Serialization.h"

#include "CondFormats/PhysicsToolsObjects/interface/Serialization.h"

int main()
{
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

    return 0;
}

#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/EgammaObjects/interface/Serialization.h"
#include "CondFormats/PhysicsToolsObjects/interface/Serialization.h"

int main()
{
    testSerialization<ElectronLikelihoodCalibration>();
    testSerialization<ElectronLikelihoodCalibration::Entry>();
    testSerialization<ElectronLikelihoodCategoryData>();
    testSerialization<GBRForest>();
    testSerialization<GBRForest2D>();
    testSerialization<GBRTree>();
    testSerialization<GBRTree2D>();
    testSerialization<std::vector<ElectronLikelihoodCalibration::Entry>>();
    testSerialization<std::vector<GBRTree2D>>();
    testSerialization<std::vector<GBRTree>>();

    return 0;
}

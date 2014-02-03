#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"


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

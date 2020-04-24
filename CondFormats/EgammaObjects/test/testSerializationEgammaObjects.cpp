#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"


int main()
{
    testSerialization<ElectronLikelihoodCalibration>();
    testSerialization<ElectronLikelihoodCalibration::Entry>();
    testSerialization<ElectronLikelihoodCategoryData>();
    testSerialization<GBRForest>();
    testSerialization<GBRForest2D>();
    testSerialization<GBRForestD>();
    testSerialization<GBRTree>();
    testSerialization<GBRTree2D>();
    testSerialization<GBRTreeD>();
    testSerialization<std::vector<ElectronLikelihoodCalibration::Entry>>();
    testSerialization<std::vector<GBRTree2D>>();
    testSerialization<std::vector<GBRTree>>();
    testSerialization<std::vector<GBRTreeD>>();

    return 0;
}

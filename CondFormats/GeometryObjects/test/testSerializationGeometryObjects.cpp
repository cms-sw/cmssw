#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<CSCRecoDigiParameters>();
    testSerialization<PCaloGeometry>();
    testSerialization<PGeometricDet>();
    //testSerialization<PGeometricDet::Item>(); has uninitialized booleans
    testSerialization<PGeometricDetExtra>();
    testSerialization<PGeometricDetExtra::Item>();
    testSerialization<RecoIdealGeometry>();
    testSerialization<std::vector<PGeometricDet::Item>>();
    testSerialization<std::vector<PGeometricDetExtra::Item>>();
    testSerialization<PTrackerParameters>();
    testSerialization<PTrackerParameters::PxbItem>();
    testSerialization<PTrackerParameters::PxfItem>();
    testSerialization<PTrackerParameters::TECItem>();
    testSerialization<PTrackerParameters::TIBItem>();
    testSerialization<PTrackerParameters::TIDItem>();
    testSerialization<PTrackerParameters::TOBItem>();

    return 0;
}

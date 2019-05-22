#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/GeometryObjects/src/headers.h"

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
    testSerialization<PTrackerParameters::Item>();
    testSerialization<HcalParameters>();
    testSerialization<PHGCalParameters>();
    testSerialization<PMTDParameters>();

    return 0;
}

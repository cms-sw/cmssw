#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/GeometryObjects/interface/Serialization.h"

int main()
{
    testSerialization<CSCRecoDigiParameters>();
    testSerialization<PCaloGeometry>();
    testSerialization<PGeometricDet>();
    testSerialization<PGeometricDet::Item>();
    testSerialization<PGeometricDetExtra>();
    testSerialization<PGeometricDetExtra::Item>();
    testSerialization<RecoIdealGeometry>();
    testSerialization<std::vector<PGeometricDet::Item>>();
    testSerialization<std::vector<PGeometricDetExtra::Item>>();

    return 0;
}

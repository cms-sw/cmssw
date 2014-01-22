
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/External/interface/DetID.h"
#include "CondFormats/External/interface/EcalDetID.h"
#include "CondFormats/External/interface/L1GtLogicParser.h"
#include "CondFormats/External/interface/SMatrix.h"
#include "CondFormats/External/interface/Timestamp.h"
#include "CondFormats/GeometryObjects/interface/Serialization.h"

#include "CondFormats/Serialization/interface/SerializationTest.h"

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

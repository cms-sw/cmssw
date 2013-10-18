#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/External/interface/Serialization.h"

int main()
{
    testSerialization<DetId>();
    testSerialization<EBDetId>();
    testSerialization<EcalContainer<EBDetId, float>>();
    testSerialization<trigger::HLTPrescaleTable>();
    testSerialization<L1GtLogicParser::TokenRPN>();
    testSerialization<edm::Timestamp>();
    testSerialization<CLHEP::Hep3Vector>();
    testSerialization<CLHEP::HepEulerAngles>();
    testSerialization<ROOT::Math::SMatrix<double, 2, 3>>();

    return 0;
}

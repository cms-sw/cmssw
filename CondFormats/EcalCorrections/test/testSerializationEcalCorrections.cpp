#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/EcalCorrections/interface/Serialization.h"

int main()
{
    testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta>();
    testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients>();
    testSerialization<EcalShowerContainmentCorrections>();
    testSerialization<EcalShowerContainmentCorrections::Coefficients>();

    return 0;
}

#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta>();
    testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients>();
    testSerialization<EcalShowerContainmentCorrections>();
    testSerialization<EcalShowerContainmentCorrections::Coefficients>();

    return 0;
}

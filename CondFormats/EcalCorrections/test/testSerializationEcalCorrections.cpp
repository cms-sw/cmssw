
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/External/interface/EcalDetID.h"
#include "CondFormats/EcalCorrections/interface/Serialization.h"

#include "CondFormats/Serialization/interface/SerializationTest.h"

int main()
{
    testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta>();
    testSerialization<EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients>();
    testSerialization<EcalShowerContainmentCorrections>();
    testSerialization<EcalShowerContainmentCorrections::Coefficients>();

    return 0;
}

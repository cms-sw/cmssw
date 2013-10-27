#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/RecoMuonObjects/interface/Serialization.h"

int main()
{
    testSerialization<MuScleFitDBobject>();

    return 0;
}

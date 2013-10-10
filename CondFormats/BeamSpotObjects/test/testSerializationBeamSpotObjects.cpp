#include "CondFormats/Common/interface/SerializationTest.h"

#include "CondFormats/BeamSpotObjects/interface/Serialization.h"

int main()
{
    testSerialization<BeamSpotObjects>();
    testSerialization<SimBeamSpotObjects>();

    return 0;
}

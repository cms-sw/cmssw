#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<BeamSpotObjects>();
    testSerialization<SimBeamSpotObjects>();

    return 0;
}

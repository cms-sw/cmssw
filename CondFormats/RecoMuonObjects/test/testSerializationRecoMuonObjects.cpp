#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<MuScleFitDBobject>();
    testSerialization<DYTThrObject>();
    return 0;
}

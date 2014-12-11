#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<MuScleFitDBobject>();
    testSerialization<DYTParamsObject>();
    testSerialization<DYTThrObject>();
    return 0;
}

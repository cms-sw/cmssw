#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<BTagEntry>();
    testSerialization<BTagEntry::Parameters>();
    testSerialization<BTagCalibration>();

    return 0;
}

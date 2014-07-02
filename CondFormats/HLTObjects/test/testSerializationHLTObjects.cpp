#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<AlCaRecoTriggerBits>();
    testSerialization<std::pair<const std::string,std::vector<unsigned int>>>();
    //testSerialization<trigger::HLTPrescaleTableCond>(); never serialized in the old DB

    return 0;
}

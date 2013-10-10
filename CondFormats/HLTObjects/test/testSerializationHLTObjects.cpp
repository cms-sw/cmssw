#include "CondFormats/Common/interface/SerializationTest.h"

#include "CondFormats/HLTObjects/interface/Serialization.h"

int main()
{
    testSerialization<AlCaRecoTriggerBits>();
    testSerialization<std::pair<const std::string,std::vector<unsigned int>>>();
    testSerialization<trigger::HLTPrescaleTableCond>();

    return 0;
}

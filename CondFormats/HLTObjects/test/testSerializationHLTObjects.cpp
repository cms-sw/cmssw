#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/External/interface/HLTPrescaleTable.h"
#include "CondFormats/HLTObjects/interface/Serialization.h"

int main()
{
    testSerialization<AlCaRecoTriggerBits>();
    testSerialization<std::pair<const std::string,std::vector<unsigned int>>>();
    //testSerialization<trigger::HLTPrescaleTableCond>(); never serialized in the old DB

    return 0;
}

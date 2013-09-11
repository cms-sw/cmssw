#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/HIObjects/interface/Serialization.h"

int main()
{
    testSerialization<CentralityTable>();
    testSerialization<CentralityTable::BinValues>();
    testSerialization<CentralityTable::CBin>();
    testSerialization<RPFlatParams>();
    testSerialization<RPFlatParams::EP>();
    testSerialization<std::vector<CentralityTable::CBin>>();
    testSerialization<std::vector<RPFlatParams::EP>>();

    return 0;
}

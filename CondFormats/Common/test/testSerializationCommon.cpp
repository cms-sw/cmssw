#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<ConfObject>();
    testSerialization<DropBoxMetadata>();
    testSerialization<DropBoxMetadata::Parameters>();
    testSerialization<FileBlob>();
    testSerialization<MultiFileBlob>();
    testSerialization<cond::BaseKeyed>();
    testSerialization<cond::GenericSummary>();
    testSerialization<cond::IOVDescription>();
    //testSerialization<cond::IOVElement>(); not used in the future
    testSerialization<cond::IOVKeysDescription>();
    testSerialization<cond::IOVProvenance>();
    //testSerialization<cond::IOVSequence>(); not used in the future
    testSerialization<cond::IOVUserMetaData>();
    testSerialization<cond::SmallWORMDict>();
    //testSerialization<cond::Summary>(); abstract
    testSerialization<cond::UpdateStamp>();
    testSerialization<cond::BasicPayload>();
    //testSerialization<ora::OId>(); not used in the future
    //testSerialization<ora::PVector<cond::IOVElement>>(); not used in the future
    //testSerialization<ora::QueryableVector<cond::IOVElement>>(); not used in the future
    testSerialization<std::map<std::string, unsigned long long>>();
    //testSerialization<std::map<const std::basic_string<char>, DropBoxMetadata::Parameters>>(); no const-key std::map template (we could provide it, but it is equivalent to a non-const key std::map, and looks unused/should be unused)
    testSerialization<std::map<std::string, DropBoxMetadata::Parameters>>();
    testSerialization<std::map<unsigned long long,unsigned long long>>();
    testSerialization<std::map<unsigned long long,unsigned long long>::value_type>();
    testSerialization<std::pair<const std::string, unsigned long long>>();
    testSerialization<std::pair<const std::basic_string<char>, DropBoxMetadata::Parameters>>();
    testSerialization<std::pair<std::string, DropBoxMetadata::Parameters>>();
    //testSerialization<std::vector<cond::IOVElement>>(); not used in the future

    return 0;
}

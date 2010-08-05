#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/GenericSummary.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/Common/interface/MultiFileBlob.h"

#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/IOVKeysDescription.h"

#include <vector>

namespace {
  namespace {
    struct dictionaries {
    };

    struct Dummy {
         std::map<unsigned long long,unsigned long long> dummyForTests;
         std::map<unsigned long long,unsigned long long>::value_type dummyForTest2;
    };

  }

}


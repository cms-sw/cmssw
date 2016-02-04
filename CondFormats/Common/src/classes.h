#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/GenericSummary.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/Common/interface/MultiFileBlob.h"

#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/IOVKeysDescription.h"
#include "CondFormats/Common/interface/ConfObject.h"

#include "CondFormats/Common/interface/DropBoxMetadata.h"


#include <vector>

namespace {
  namespace {
    struct dictionaries {
    };

    struct Dummy {
         std::map<unsigned long long,unsigned long long> dummyForTests;
         std::map<unsigned long long,unsigned long long>::value_type dummyForTest2;

      DropBoxMetadata::Parameters aparam;
      std::pair<std::string, DropBoxMetadata::Parameters> apair1;
      std::pair<const std::basic_string<char>, DropBoxMetadata::Parameters> apair2;

      std::map<std::string, DropBoxMetadata::Parameters> amap1;
      std::map<const std::basic_string<char>, DropBoxMetadata::Parameters> amap2;

    };

  }

}


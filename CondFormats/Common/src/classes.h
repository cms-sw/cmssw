#include "CondFormats/Common/src/headers.h"


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


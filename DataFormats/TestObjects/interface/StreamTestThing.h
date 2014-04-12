#ifndef DataFormats_TestObjects_StreamTestThing_h
#define DataFormats_TestObjects_StreamTestThing_h

#include "FWCore/Utilities/interface/typedefs.h"

#include <vector>

namespace edmtestprod {

  struct StreamTestThing {
    ~StreamTestThing();
    explicit StreamTestThing(int sz);
    StreamTestThing();

    std::vector<cms_int32_t> data_;
  };

}

#endif

#ifndef DataFormats_TestObjects_Thing_h
#define DataFormats_TestObjects_Thing_h

#include "FWCore/Utilities/interface/typedefs.h"

namespace edmtest {

  struct Thing {
    ~Thing() { }
    Thing():a() { }
    cms_int32_t a;
  };

}

#endif

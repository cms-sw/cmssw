#ifndef DataFormats_TestObjects_ThingWithMerge_h
#define DataFormats_TestObjects_ThingWithMerge_h

#include "FWCore/Utilities/interface/typedefs.h"

namespace edmtest {

  struct ThingWithMerge {
    ~ThingWithMerge() { }
    ThingWithMerge():a() { }
    bool mergeProduct(ThingWithMerge const& newThing);
    cms_int32_t a;
  };

}

#endif

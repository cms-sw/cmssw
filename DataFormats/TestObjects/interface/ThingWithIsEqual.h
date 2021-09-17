#ifndef DataFormats_TestObjects_ThingWithIsEqual_h
#define DataFormats_TestObjects_ThingWithIsEqual_h

#include "FWCore/Utilities/interface/typedefs.h"

namespace edmtest {

  struct ThingWithIsEqual {
    ~ThingWithIsEqual() {}
    ThingWithIsEqual() : a() {}
    explicit ThingWithIsEqual(cms_int32_t v) : a(v) {}
    bool isProductEqual(ThingWithIsEqual const& thingWithIsEqual) const;
    cms_int32_t a;
  };

}  // namespace edmtest

#endif

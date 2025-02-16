#ifndef DataFormats_TestObjects_interface_ThingWithPostInsert_h
#define DataFormats_TestObjects_interface_ThingWithPostInsert_h

#include <cstdint>

#include "FWCore/Utilities/interface/typedefs.h"

namespace edmtest {

  class ThingWithPostInsert {
  public:
    ThingWithPostInsert() : value_{0}, valid_{false} {};
    explicit ThingWithPostInsert(cms_int32_t v) : value_{v}, valid_{false} {}

    void post_insert() { valid_ = true; }

    int32_t value() const { return value_; }

    bool valid() const { return valid_; }

  private:
    cms_int32_t value_;
    bool valid_;
  };

}  // namespace edmtest

#endif  // DataFormats_TestObjects_interface_ThingWithPostInsert_h

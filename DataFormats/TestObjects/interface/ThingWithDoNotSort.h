#ifndef DataFormats_TestObjects_interface_ThingWithDoNotSort_h
#define DataFormats_TestObjects_interface_ThingWithDoNotSort_h

#include <stdexcept>
#include <cstdint>

#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/typedefs.h"

namespace edmtest {

  class ThingWithDoNotSort : public edm::DoNotSortUponInsertion {
  public:
    ThingWithDoNotSort() : value_{0} {};
    explicit ThingWithDoNotSort(cms_int32_t v) : value_{v} {}

    void post_insert() {
      throw cms::Exception("LogicError")
          << "post_insert() called for ThingWithDoNotSort that inherits from edm::DoNotSortUponInsertion";
    }

    int32_t value() const { return value_; }

  private:
    cms_int32_t value_;
  };

}  // namespace edmtest

#endif  // DataFormats_TestObjects_interface_ThingWithDoNotSort_h

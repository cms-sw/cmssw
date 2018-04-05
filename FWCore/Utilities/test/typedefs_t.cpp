#include "FWCore/Utilities/interface/typedefs.h"

// Will fail compilation if any assert is violated.
int main() {
   static_assert(sizeof(cms_int8_t) == 1,"type does not match size problem");
   static_assert(sizeof(cms_uint8_t) == 1,"type does not match size problem");
   static_assert(sizeof(cms_int16_t) == 2,"type does not match size problem");
   static_assert(sizeof(cms_uint16_t) == 2,"type does not match size problem");
   static_assert(sizeof(cms_int32_t) == 4,"type does not match size problem");
   static_assert(sizeof(cms_uint32_t) == 4,"type does not match size problem");
   static_assert(sizeof(cms_int64_t) == 8,"type does not match size problem");
   static_assert(sizeof(cms_uint64_t) == 8,"type does not match size problem");
}

#include "FWCore/Utilities/interface/typedefs.h"

#include "boost/static_assert.hpp"

// Will fail compilation if any assert is violated.
int main() {
   BOOST_STATIC_ASSERT(sizeof(cms_int8_t) == 1);
   BOOST_STATIC_ASSERT(sizeof(cms_uint8_t) == 1);
   BOOST_STATIC_ASSERT(sizeof(cms_int16_t) == 2);
   BOOST_STATIC_ASSERT(sizeof(cms_uint16_t) == 2);
   BOOST_STATIC_ASSERT(sizeof(cms_int32_t) == 4);
   BOOST_STATIC_ASSERT(sizeof(cms_uint32_t) == 4);
   BOOST_STATIC_ASSERT(sizeof(cms_int64_t) == 8);
   BOOST_STATIC_ASSERT(sizeof(cms_uint64_t) == 8);
}

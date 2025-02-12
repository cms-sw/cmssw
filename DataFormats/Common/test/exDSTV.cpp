#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <functional>
#include "catch.hpp"

namespace {
  struct T {
    float v;
  };
}  // namespace

typedef edmNew::DetSetVector<T> DSTV;
typedef edmNew::DetSet<T> DST;
typedef DSTV::FastFiller FF;

TEST_CASE("test edmNew::DetSetVector", "[edmNew::DetSetVector]") {
  DSTV dstv;

  {
    FF ff(dstv, 1);
    ff.push_back(T());
    ff[0].v = 2.1f;
  }
  {
    FF ff(dstv, 2);
    ff.push_back(T());
    ff.push_back(T());
  }
  REQUIRE(dstv.size() == 2);
  REQUIRE(dstv.dataSize() == 3);
  REQUIRE(dstv.detsetSize(0) == 1);

  DST d1 = *dstv.find(2);
  d1[0].v = 3.14f;
  DST d2 = dstv.insert(4, 3);
  d2[0].v = 4.15f;

  SECTION("iterator") {
    int i = 0;
    std::vector<float> values = {2.1, 3.14, 4.15};
    std::for_each(dstv.begin(), dstv.end(), [&i, &values](auto const& value) { REQUIRE(value[0].v == values[i++]); });
  }
}

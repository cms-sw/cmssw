/**
   \file
   test file for FEDRawData library

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include "catch2/catch_all.hpp"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>

#include <iostream>

TEST_CASE("FEDRawData", "[FEDRawData]") {
  SECTION("testCtor") {
    FEDRawData f;
    REQUIRE(f.size() == 0);

    FEDRawData f2(24);
    REQUIRE(f2.size() == size_t(24));
  }

  SECTION("testdata") {
    FEDRawData f(48);
    f.data()[0] = 'a';
    f.data()[1] = 'b';
    f.data()[47] = 'c';

    const unsigned char* buf = f.data();

    REQUIRE(buf[0] == 'a');
    REQUIRE(buf[1] == 'b');
    REQUIRE(buf[47] == 'c');
  }
}

/**
   \file
   unit test file for class FEDRawDataProduct 

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include "catch2/catch_all.hpp"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

TEST_CASE("RawDataBufferProduct", "[FEDRawData]") {
  SECTION("testInsertAndReadBack") {
    FEDRawData f1(16);
    f1.data()[0] = 'a';
    f1.data()[1] = 'b';

    FEDRawData f2(24);
    f2.data()[0] = 'd';
    f2.data()[1] = 'e';

    FEDRawDataCollection fp;
    fp.FEDData(12) = f1;
    fp.FEDData(121) = f2;

    REQUIRE(fp.FEDData(12).data()[0] == 'a');
    REQUIRE(fp.FEDData(12).data()[1] == 'b');

    REQUIRE(fp.FEDData(121).data()[0] == 'd');
    REQUIRE(fp.FEDData(121).data()[1] == 'e');
  }
}

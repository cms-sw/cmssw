/**
   \file
   test file for FEDRawData library

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include "catch2/catch_all.hpp"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

TEST_CASE("FEDNumbering", "[FEDRawData]") {
  SECTION("test_inRange") {
    int i = 0;
    for (i = FEDNumbering::MINSiPixelFEDID; i <= FEDNumbering::MAXSiPixelFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MAXSiPixelFEDID + 1; i <= FEDNumbering::MINSiStripFEDID - 1; i++) {
      REQUIRE(not FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINSiStripFEDID; i <= FEDNumbering::MAXSiStripFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINPreShowerFEDID; i <= FEDNumbering::MAXPreShowerFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINECALFEDID; i <= FEDNumbering::MAXECALFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINCASTORFEDID; i <= FEDNumbering::MAXCASTORFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINLUMISCALERSFEDID; i <= FEDNumbering::MAXLUMISCALERSFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINCSCFEDID; i <= FEDNumbering::MAXCSCFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINCSCTFFEDID; i <= FEDNumbering::MAXCSCTFFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINDTFEDID; i <= FEDNumbering::MAXDTFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINDTTFFEDID; i <= FEDNumbering::MAXDTTFFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINRPCFEDID; i <= FEDNumbering::MAXRPCFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTriggerGTPFEDID; i <= FEDNumbering::MAXTriggerGTPFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTriggerEGTPFEDID; i <= FEDNumbering::MAXTriggerEGTPFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTriggerGCTFEDID; i <= FEDNumbering::MAXTriggerGCTFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTriggerLTCFEDID; i <= FEDNumbering::MAXTriggerLTCFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTriggerLTCmtccFEDID; i <= FEDNumbering::MAXTriggerLTCmtccFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINCSCDDUFEDID; i <= FEDNumbering::MAXCSCDDUFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINCSCContingencyFEDID; i <= FEDNumbering::MAXCSCContingencyFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINCSCTFSPFEDID; i <= FEDNumbering::MAXCSCTFSPFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINDAQeFEDFEDID; i <= FEDNumbering::MAXDAQeFEDFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINDAQmFEDFEDID; i <= FEDNumbering::MAXDAQmFEDFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTCDSuTCAFEDID; i <= FEDNumbering::MAXTCDSuTCAFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINHCALuTCAFEDID; i <= FEDNumbering::MAXHCALuTCAFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINSiPixeluTCAFEDID; i <= FEDNumbering::MAXSiPixeluTCAFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINDTUROSFEDID; i <= FEDNumbering::MAXDTUROSFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
    for (i = FEDNumbering::MINTriggerUpgradeFEDID; i <= FEDNumbering::MAXTriggerUpgradeFEDID; i++) {
      REQUIRE(FEDNumbering::inRange(i));
    }
  }
}

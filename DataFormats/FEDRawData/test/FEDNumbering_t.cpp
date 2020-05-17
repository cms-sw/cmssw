/**
   \file
   test file for FEDRawData library

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

class testFEDNumbering : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFEDNumbering);

  CPPUNIT_TEST(test_inRange);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void test_inRange();
  void test_fromDet();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testFEDNumbering);

void testFEDNumbering::test_inRange() {
  int i = 0;
  for (i = FEDNumbering::MINSiPixelFEDID; i <= FEDNumbering::MAXSiPixelFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MAXSiPixelFEDID + 1; i <= FEDNumbering::MINSiStripFEDID - 1; i++) {
    CPPUNIT_ASSERT(not FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINSiStripFEDID; i <= FEDNumbering::MAXSiStripFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINPreShowerFEDID; i <= FEDNumbering::MAXPreShowerFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINECALFEDID; i <= FEDNumbering::MAXECALFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINCASTORFEDID; i <= FEDNumbering::MAXCASTORFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINLUMISCALERSFEDID; i <= FEDNumbering::MAXLUMISCALERSFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINCSCFEDID; i <= FEDNumbering::MAXCSCFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINCSCTFFEDID; i <= FEDNumbering::MAXCSCTFFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINDTFEDID; i <= FEDNumbering::MAXDTFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINDTTFFEDID; i <= FEDNumbering::MAXDTTFFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINRPCFEDID; i <= FEDNumbering::MAXRPCFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTriggerGTPFEDID; i <= FEDNumbering::MAXTriggerGTPFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTriggerEGTPFEDID; i <= FEDNumbering::MAXTriggerEGTPFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTriggerGCTFEDID; i <= FEDNumbering::MAXTriggerGCTFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTriggerLTCFEDID; i <= FEDNumbering::MAXTriggerLTCFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTriggerLTCmtccFEDID; i <= FEDNumbering::MAXTriggerLTCmtccFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINCSCDDUFEDID; i <= FEDNumbering::MAXCSCDDUFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINCSCContingencyFEDID; i <= FEDNumbering::MAXCSCContingencyFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINCSCTFSPFEDID; i <= FEDNumbering::MAXCSCTFSPFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINDAQeFEDFEDID; i <= FEDNumbering::MAXDAQeFEDFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINDAQmFEDFEDID; i <= FEDNumbering::MAXDAQmFEDFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTCDSuTCAFEDID; i <= FEDNumbering::MAXTCDSuTCAFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINHCALuTCAFEDID; i <= FEDNumbering::MAXHCALuTCAFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINSiPixeluTCAFEDID; i <= FEDNumbering::MAXSiPixeluTCAFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINDTUROSFEDID; i <= FEDNumbering::MAXDTUROSFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
  for (i = FEDNumbering::MINTriggerUpgradeFEDID; i <= FEDNumbering::MAXTriggerUpgradeFEDID; i++) {
    CPPUNIT_ASSERT(FEDNumbering::inRange(i));
  }
}

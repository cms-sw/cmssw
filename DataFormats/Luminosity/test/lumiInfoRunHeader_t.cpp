#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

class TestLumiInfoRunHeader : public CppUnit::TestFixture {
  static const float tol;

  CPPUNIT_TEST_SUITE(TestLumiInfoRunHeader);
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testConstructor();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestLumiInfoRunHeader);

void TestLumiInfoRunHeader::testConstructor() {
  std::cout << "\nTesting LumiInfoRunHeader\n";

  std::string lumiProviderName = "lumiProviderTest";
  std::string fillingSchemeName = "fillingSchemeTest";
  std::bitset<LumiConstants::numBX> fillingScheme;

  fillingScheme[1] = true;
  fillingScheme[10] = true;
  fillingScheme[12] = true;
  fillingScheme[16] = true;

  LumiInfoRunHeader lumiInfoRH(lumiProviderName, fillingSchemeName, fillingScheme);

  CPPUNIT_ASSERT(lumiInfoRH.getLumiProvider() == lumiProviderName);
  CPPUNIT_ASSERT(lumiInfoRH.getFillingSchemeName() == fillingSchemeName);
  CPPUNIT_ASSERT(lumiInfoRH.getFillingScheme() == fillingScheme);
  CPPUNIT_ASSERT(lumiInfoRH.getBunchFilled(0) == false);
  CPPUNIT_ASSERT(lumiInfoRH.getBunchFilled(1) == true);
  CPPUNIT_ASSERT(lumiInfoRH.getBunchSpacing() == 50);
}

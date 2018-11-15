#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDStrVector.h"
#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

class testDDStrVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDStrVector);
  CPPUNIT_TEST(checkAgaistOld);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() override{}
  void tearDown() override {}
  void buildIt();
  void testloading();
  void checkAgaistOld();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDStrVector);

void
testDDStrVector::buildIt() {
  auto strVec = std::make_unique< std::vector< std::string >>();
  strVec->emplace_back("One");
  strVec->emplace_back("Two");
  strVec->emplace_back("Three");
  
  DDStrVector testVec( "TestVector", std::move( strVec ));
  std::cerr << testVec << std::endl;
}

void
testDDStrVector::testloading() {
  auto strVec = std::make_unique< std::vector< std::string >>();
  strVec->emplace_back("One");
  strVec->emplace_back("Two");
  strVec->emplace_back("Three");
  
  DDStrVector testVec( "TestVector", std::move( strVec ));
  std::ostringstream  os;
  os << testVec;
  std::string str( "DDStrVector name=GLOBAL:TestVector size=3 vals=( One Two Three )" );
  if( os.str() != str ) std::cerr << "not the same!" << std::endl;
  CPPUNIT_ASSERT( os.str() == str );
}

void
testDDStrVector::checkAgaistOld() {
  buildIt();
  testloading();
}

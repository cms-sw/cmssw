#include "DetectorDescription/Core/interface/DDStrVector.h"
#include <cppunit/extensions/HelperMacros.h>

class testDDStrVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDStrVector);
  CPPUNIT_TEST(checkAgaistOld);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown() {}
  void buildIt();
  void testloading();
  void checkAgaistOld();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDStrVector);

void
testDDStrVector::buildIt() {
  auto strVec = new std::vector<std::string>;
  strVec->push_back("One");
  strVec->push_back("Two");
  strVec->push_back("Three");
  
  DDStrVector testVec( "TestVector", strVec );
  std::cerr << testVec << std::endl;
}

void
testDDStrVector::testloading() {
  auto strVec = new std::vector<std::string>;
  strVec->push_back("One");
  strVec->push_back("Two");
  strVec->push_back("Three");
  
  DDStrVector testVec( "TestVector", strVec );
  std::ostringstream  os;
  os << testVec;
  std::string str("DDStrVector name=GLOBAL:TestVector size=3 vals=( One Two Three )");
  if (os.str()!=str) std::cerr << "not the same!" << std::endl;
  CPPUNIT_ASSERT (os.str()==str);
}

void
testDDStrVector::checkAgaistOld() {
  buildIt();
  testloading();
}

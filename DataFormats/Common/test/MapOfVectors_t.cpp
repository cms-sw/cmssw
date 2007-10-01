#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "DataFormats/Common/interface/MapOfVectors.h"


#undef private

#include "FWCore/Utilities/interface/EDMException.h"

#include<vector>
#include<algorithm>

class TestMapOfVectors: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestMapOfVectors);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(filling);
  CPPUNIT_TEST(find);
  CPPUNIT_TEST(iterator);

  
  CPPUNIT_TEST_SUITE_END();
  
public:
  TestMapOfVectors(); 
  ~TestMapOfVectors() 
  void setUp() {}
  void tearDown() {}
  
  void default_ctor();
  void filling();
  void find();
  void iterator();
  

public:

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDetSet);

TestMapOfVectors::TestMapOfVectors(){} 
TestMapOfVectors::~TestMapOfVectors() {}

  
void TestMapOfVectors::default_ctor(){}
void TestMapOfVectors::filling(){}
void TestMapOfVectors::find(){}
void TestMapOfVectors::iterator(){}

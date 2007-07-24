#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"
#undef private

class TestDataFrame: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestDataFrame);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(filling);
  CPPUNIT_TEST(iterator);

  CPPUNIT_TEST_SUITE_END();

 public:
  TestDataFrame() { } 
  ~TestDataFrame() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor();
  void filling();
  void iterator();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDataFrame);

void TestDataFrame::default_ctor() {

  edm::DataFrameContainer frames(10,2);
  CPPUNIT_ASSERT(frames.stride()==10);
  CPPUNIT_ASSERT(frames.subdetId()==2);
  CPPUNIT_ASSERT(frames.size()==0);
  frames.resize(2);
  CPPUNIT_ASSERT(frames.size()==2);
  edm::DataFrame df = frames[1];
  CPPUNIT_ASSERT(df.size()==10); 
  CPPUNIT_ASSERT(df.m_data==&frames.m_data.front()+10); 
}

void TestDataFrame::filling() {


}

void TestDataFrame::iterator() {


}



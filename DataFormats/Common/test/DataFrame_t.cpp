#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"
#undef private
#include<vector>
#include<algorithm>
#include<cstdlib>

class TestDataFrame: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestDataFrame);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(filling);
  CPPUNIT_TEST(iterator);

  CPPUNIT_TEST_SUITE_END();

 public:
  TestDataFrame(); 
  ~TestDataFrame() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor();
  void filling();
  void iterator();

public:
  std::vector<edm::DataFrame::data_type> sv;

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDataFrame);

TestDataFrame::TestDataFrame() : sv(10){ 
  edm::DataFrame::data_type v[10] = {0,1,2,3,4,5,6,7,8,9};
  std::copy(v,v+10,sv.begin());
} 


void TestDataFrame::default_ctor() {

  edm::DataFrameContainer frames(10,2);
  CPPUNIT_ASSERT(frames.stride()==10);
  CPPUNIT_ASSERT(frames.subdetId()==2);
  CPPUNIT_ASSERT(frames.size()==0);
  frames.resize(3);
  CPPUNIT_ASSERT(frames.size()==3);
  edm::DataFrame df = frames[1];
  CPPUNIT_ASSERT(df.size()==10); 
  CPPUNIT_ASSERT(df.m_data==&frames.m_data.front()+10); 
  df.set(frames,2);
  CPPUNIT_ASSERT(df.size()==10); 
  CPPUNIT_ASSERT(df.m_data==&frames.m_data.front()+20); 
}

void TestDataFrame::filling() {

  edm::DataFrameContainer frames(10,2);
  for (int n=1;n<5;++n) {
    int id=20+n;
    frames.push_back(id);
    CPPUNIT_ASSERT(frames.size()==n);
    edm::DataFrame df = frames.back();
    CPPUNIT_ASSERT(df.size()==10); 
    CPPUNIT_ASSERT(df.id()==id); 
    
    if (n%2==0)
      std::copy(sv.begin(),sv.end(),df.begin());  
    else
      ::memcpy(&sv[0],&df[0], sizeof(edm::DataFrame::data_type)*frames.stride());
    
    std::vector<edm::DataFrame::data_type> v2(10);
    std::copy(frames.m_data.begin()+(n-1)*10,frames.m_data.begin()+n*10,v2.begin());
    CPPUNIT_ASSERT(sv==v2);
  }

}

namespace {
  struct VerifyIter{
    VerifyIter(TestDataFrame * itest):n(0), test(*itest){}
    
    void operator()(edm::DataFrame const & df) {
      ++n;
      CPPUNIT_ASSERT(df.id()==20+n); 
      std::vector<edm::DataFrame::data_type> v2(10);
      std::copy(df.begin(),df.end(),v2.begin());
      CPPUNIT_ASSERT(test.sv==v2);
    }
    
    int n;
  TestDataFrame & test;
  };
}

void TestDataFrame::iterator() {
  edm::DataFrameContainer frames(10,2);
  for (int n=1;n<5;++n) {
    int id=20+n;
    frames.push_back(id);
    edm::DataFrame df = frames.back();
    std::copy(sv.begin(),sv.end(),df.begin());
  }
  CPPUNIT_ASSERT(std::for_each(frames.begin(),frames.end(),VerifyIter(this)).n==4);


}



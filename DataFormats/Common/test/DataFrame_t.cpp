#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#define private public
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"
#undef private
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <boost/bind.hpp>
#include <numeric>
#include <cstring>

class TestDataFrame: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestDataFrame);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(filling);
  CPPUNIT_TEST(iterator);
  CPPUNIT_TEST(sort);

  CPPUNIT_TEST_SUITE_END();

 public:
  TestDataFrame(); 
  ~TestDataFrame() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor();
  void filling();
  void iterator();
  void sort();

public:
  std::vector<edm::DataFrame::data_type> sv1;
  std::vector<edm::DataFrame::data_type> sv2;

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDataFrame);

TestDataFrame::TestDataFrame() : sv1(10),sv2(10){ 
  edm::DataFrame::data_type v[10] = {0,1,2,3,4,5,6,7,8,9};
  std::copy(v,v+10,sv1.begin());
  std::transform(sv1.begin(),sv1.end(),sv2.begin(),boost::bind(std::plus<edm::DataFrame::data_type>(),10,_1));
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
  frames.pop_back();
  CPPUNIT_ASSERT(frames.size()==2);
  CPPUNIT_ASSERT(frames.m_data.size()==20);
  
}

void TestDataFrame::filling() {

  edm::DataFrameContainer frames(10,2);
  for (unsigned int n=1;n<5;++n) {
    unsigned int id=20+n;
    frames.push_back(id);
    CPPUNIT_ASSERT(frames.size()==n);
    edm::DataFrame df = frames.back();
    CPPUNIT_ASSERT(df.size()==10); 
    CPPUNIT_ASSERT(df.id()==id); 
    
    if (n%2==0)
      std::copy(sv1.begin(),sv1.end(),df.begin());  
    else
      ::memcpy(&df[0], &sv1[0], sizeof(edm::DataFrame::data_type)*frames.stride());
    
    std::vector<edm::DataFrame::data_type> v2(10);
    std::copy(frames.m_data.begin()+(n-1)*10,frames.m_data.begin()+n*10,v2.begin());
    CPPUNIT_ASSERT(sv1==v2);
  }

}

namespace {
  struct VerifyIter{
    VerifyIter(TestDataFrame * itest):n(0), test(*itest){}
    
    void operator()(edm::DataFrame const & df) {
      ++n;
      CPPUNIT_ASSERT(df.id()==2000+n); 
      std::vector<edm::DataFrame::data_type> v2(10);
      std::copy(df.begin(),df.end(),v2.begin());
      if (n%2==0) 
	CPPUNIT_ASSERT(test.sv1==v2);
      else
	CPPUNIT_ASSERT(test.sv2==v2);
    }
    
    unsigned int n;
  TestDataFrame & test;
  };
}

void TestDataFrame::iterator() {
  edm::DataFrameContainer frames(10,2);
  for (int n=1;n<5;++n) {
    int id=2000+n;
    frames.push_back(id);
    edm::DataFrame df = frames.back();
    if (n%2==0) 
      std::copy(sv1.begin(),sv1.end(),df.begin());
    else
      std::copy(sv2.begin(),sv2.end(),df.begin());
  }
  CPPUNIT_ASSERT(std::for_each(frames.begin(),frames.end(),VerifyIter(this)).n==4);
}


void TestDataFrame::sort() {
  edm::DataFrameContainer frames(10,2);
  std::vector<unsigned int> ids(100,1);
  ids[0]=2001;
  std::partial_sum(ids.begin(),ids.end(),ids.begin());
  std::random_shuffle(ids.begin(),ids.end());

  for (int n=0;n<100;++n) {
    frames.push_back(ids[n]);
    edm::DataFrame df = frames.back();
    if (ids[n]%2==0) 
      std::copy(sv1.begin(),sv1.end(),df.begin());
    else
      std::copy(sv2.begin(),sv2.end(),df.begin());
  }
  frames.sort();
  CPPUNIT_ASSERT(std::for_each(frames.begin(),frames.end(),VerifyIter(this)).n==100);
}



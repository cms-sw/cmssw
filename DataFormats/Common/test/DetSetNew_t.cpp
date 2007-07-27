#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#undef private
#include<vector>
#include<algorithm>

struct T {
  T(float iv=0) : v(iv){}
  float v;
};

typedef edmNew::DetSetVector<T> DSTV;
typedef edmNew::DetSet<T> DST;
typedef DSTV::FastFiller FF;


class TestDetSet: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestDetSet);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(inserting);
  CPPUNIT_TEST(filling);
  CPPUNIT_TEST(iterator);

  CPPUNIT_TEST_SUITE_END();

 public:
  TestDetSet(); 
  ~TestDetSet() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor();
  void inserting();
  void filling();
  void iterator();

public:
  std::vector<DSTV::data_type> sv;

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDetSet);

TestDetSet::TestDetSet() : sv(10){ 
  DSTV::data_type v[10] = {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.};
  std::copy(v,v+10,sv.begin());
} 


void TestDetSet::default_ctor() {

  DSTV detsets(2);
  CPPUNIT_ASSERT(detsets.subdetId()==2);
  CPPUNIT_ASSERT(detsets.size()==0);
  CPPUNIT_ASSERT(detsets.dataSize()==0);
  detsets.resize(3,10);
  CPPUNIT_ASSERT(detsets.size()==3);
  CPPUNIT_ASSERT(detsets.dataSize()==10);
  // follow is nonsense still valid construct...
  DST df(detsets,1);
  CPPUNIT_ASSERT(df.size()==0); 
  CPPUNIT_ASSERT(df.m_data==&detsets.m_data.front()); 
  df.set(detsets,2);
  CPPUNIT_ASSERT(df.size()==0); 
  CPPUNIT_ASSERT(df.m_data==&detsets.m_data.front()); 
}

void TestDetSet::inserting() {

  DSTV detsets(2);
  unsigned int ntot=0;
  for (unsigned int n=1;n<5;++n) {
    ntot+=n;
    unsigned int id=20+n;
    DST df = detsets.insert(id,n);
    CPPUNIT_ASSERT(detsets.size()==n);
    CPPUNIT_ASSERT(detsets.dataSize()==ntot);
    CPPUNIT_ASSERT(detsets.detsetSize(n-1)==n);
    CPPUNIT_ASSERT(df.size()==n); 
    CPPUNIT_ASSERT(df.id()==id); 
    
    std::copy(sv.begin(),sv.begin()+n,df.begin());

    std::vector<DST::data_type> v1(n);
    std::vector<DST::data_type> v2(n);
    std::copy(detsets.m_data.begin()+ntot-n,detsets.m_data.begin()+ntot,v2.begin());
    std::copy(sv.begin(),sv.begin()+n,v1);
    CPPUNIT_ASSERT(v1==v2);
  }

}
void TestDetSet::filling() {

  DSTV detsets(2);
  unsigned int ntot=0;
  for (unsigned int n=1;n<5;++n) {
    unsigned int id=20+n;
    FF ff(detsets, id);
    CPPUNIT_ASSERT(detsets.size()==n);
    CPPUNIT_ASSERT(detsets.dataSize()==ntot);
    CPPUNIT_ASSERT(detsets.detsetSize(n-1)==0);
    CPPUNIT_ASSERT(ff.item.offset==detsets.dataSize()); 
    CPPUNIT_ASSERT(ff.item.size==0); 
    CPPUNIT_ASSERT(ff.item.id==id); 
    ntot+=1;
    ff.push_back(3.14);
    CPPUNIT_ASSERT(detsets.dataSize()==ntot);
    CPPUNIT_ASSERT(detsets.detsetSize(n-1)==1);
    CPPUNIT_ASSERT(detsets.m_data.back().v==3.14);
    CPPUNIT_ASSERT(ff.item.offset==detsets.size()-1); 
    CPPUNIT_ASSERT(ff.item.size==1);  
    ntot+=n-1;
    ff.resize(ntot);
    CPPUNIT_ASSERT(detsets.dataSize()==ntot);
    CPPUNIT_ASSERT(detsets.detsetSize(n-1)==n);
    CPPUNIT_ASSERT(ff.item.offset==detsets.dataSize()-n); 
    CPPUNIT_ASSERT(ff.item.size==n);  
    
    std::copy(sv.begin(),sv.begin()+n,ff.begin());

    std::vector<DST::data_type> v1(n);
    std::vector<DST::data_type> v2(n);
    std::copy(detsets.m_data.begin()+ntot-n,detsets.m_data.begin()+ntot,v2.begin());
    std::copy(sv.begin(),sv.begin()+n,v1);
    CPPUNIT_ASSERT(v1==v2);
  }

}

namespace {
  struct VerifyIter{
    VerifyIter(TestDetSet * itest):n(0), test(*itest){}
    
    void operator()(DST const & df) {
      ++n;
      CPPUNIT_ASSERT(df.id()==20+n); 
      std::vector<DST::data_type> v1(n);
      std::vector<DST::data_type> v2(n);
      std::copy(df.begin(),df.end(),v2.begin());
      std::copy(test.sv.begin(),test.sv.begin()+n,v1);
      CPPUNIT_ASSERT(v1==v2);
    }
    
    unsigned int n;
    TestDetSet & test;
  };
}

void TestDetSet::iterator() {
  DSTV detsets(2);
  for (unsigned int n=1;n<5;++n) {
    unsigned int id=20+n;
    FF ff(detsets,id);
    ff.resize(n);
    std::copy(sv.begin(),sv.begin()+n,ff.begin());
  }
  CPPUNIT_ASSERT(std::for_each(detsets.begin(),detsets.end(),VerifyIter(this)).n==4);


}

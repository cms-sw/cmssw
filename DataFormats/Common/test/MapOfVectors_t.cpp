#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "DataFormats/Common/interface/MapOfVectors.h"
#undef private

#include "FWCore/Utilities/interface/EDMException.h"

#include<vector>
#include<algorithm>

typedef edm::MapOfVectors<int,int>  MII;
typedef MII::TheMap TheMap;

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
  ~TestMapOfVectors();
  void setUp() {}
  void tearDown() {}
  
  void default_ctor();
  void filling();
  void find();
  void iterator();
  
  TheMap om;
  unsigned int tot;
public:

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMapOfVectors);

TestMapOfVectors::TestMapOfVectors()  {
  int v[10] = {0,1,2,3,4,5,6,7,8,9};
  tot=0;
  for (int i=0;i<10;++i) {
    tot+=i;
    om[i].resize(i);
    std::copy(v,v+i,om[i].begin());
  }

}
 
TestMapOfVectors::~TestMapOfVectors() {}

  
void TestMapOfVectors::default_ctor(){
  MII m;
  CPPUNIT_ASSERT(m.size()==0);
  CPPUNIT_ASSERT(m.empty());
}

void TestMapOfVectors::filling(){
  MII m(om);
  CPPUNIT_ASSERT(m.size()==om.size());
  CPPUNIT_ASSERT(!m.empty());
  CPPUNIT_ASSERT(m.m_keys.size()==om.size());
  CPPUNIT_ASSERT(m.m_offsets.size()==om.size()+1);
  CPPUNIT_ASSERT(m.m_data.size()==tot);
}

void TestMapOfVectors::find(){
  MII m(om);
  CPPUNIT_ASSERT(m.find(-1)==m.emptyRange());
  for(TheMap::const_iterator p=om.begin(); p!=om.end();++p) {
    MII::range r = m.find((*p).first); 
    CPPUNIT_ASSERT(std::equal((*p).second.begin(), (*p).second.end(),r.begin()));
  }
}

void TestMapOfVectors::iterator(){
  MII m(om);
 for(MII::const_iterator p=m.begin(); p!=m.end();++p) {
 }
}

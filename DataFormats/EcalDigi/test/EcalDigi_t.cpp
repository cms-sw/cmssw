#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"
#undef private
#include<vector>
#include<algorithm>

template<typename DigiCollection>
class TestEcalDigi: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestEcalDigi);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(filling);
  CPPUNIT_TEST(iterator);

  CPPUNIT_TEST_SUITE_END();

 public:
  TestEcalDigi(); 
  ~TestEcalDigi() {}
  void setUp() {}
  void tearDown() {}
  
  void default_ctor();
  void filling();
  void iterator();

public:
  std::vector<edm::DataFrame::data_type> sv;

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestEcalDigi<EBDigiCollection>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestEcalDigi<EEDigiCollection>);

template<typename DigiCollection>
TestEcalDigi<DigiCollection>::TestEcalDigi() : sv(10){ 
  edm::DataFrame::data_type v[10] = {0,1,2,3,4,5,6,7,8,9};
  std::copy(v,v+10,sv.begin());
} 

namespace {

  void check_ctor(EBDigiCollection const& digis) {
    CPPUNIT_ASSERT(digis.subdetId()==EcalBarrel);
  }
  void check_ctor(EEDigiCollection const& digis) {
    CPPUNIT_ASSERT(digis.subdetId()==EcalEndcap);
  }


}


template<typename DigiCollection>
void TestEcalDigi<DigiCollection>::default_ctor() {

  DigiCollection frames;
  CPPUNIT_ASSERT(frames.stride()==10);
  // against enum in ExDetID
  CPPUNIT_ASSERT(frames.subdetId()==DigiCollection::DetId::Subdet);
  // against static metod in ExDetID
  CPPUNIT_ASSERT(frames.subdetId()==DigiCollection::DetId::subdet());
}

template<typename DigiCollection>
void TestEcalDigi<DigiCollection>::filling() {
  
  DigiCollection frames;
  for (int n=1;n<5;++n) {
    typename DigiCollection::DetId id = DigiCollection::DetId::unhashIndex(n);
    frames.push_back(id);
    CPPUNIT_ASSERT(int(frames.size())==n);
    typename DigiCollection::Digi df(frames.back());
    CPPUNIT_ASSERT(df.size()==10); 
    CPPUNIT_ASSERT(df.id()==id); 
    if (n%2==0) {
      // modern way of filling
      std::copy(sv.begin(),sv.end(),df.frame().begin());
      
      std::vector<edm::DataFrame::data_type> v2(10);
      std::copy(frames.m_data.begin()+(n-1)*10,frames.m_data.begin()+n*10,v2.begin());
      CPPUNIT_ASSERT(sv==v2);
    } else {
      // classical way of filling
      for (int i=0; i<10; i++)
	df.setSample(i,sv[i]);
      
      std::vector<edm::DataFrame::data_type> v2(10);
      std::copy(frames.m_data.begin()+(n-1)*10,frames.m_data.begin()+n*10,v2.begin());
      CPPUNIT_ASSERT(sv==v2);
    }
  }
  
}

namespace {
  template<typename Digi>
  struct VerifyIter{
    VerifyIter( std::vector<edm::DataFrame::data_type> const & v):n(0), sv(v){}
    void operator()(Digi const & df) {
      typedef typename Digi::key_type DetId;
      ++n;
      DetId id = DetId::unhashIndex(n);
      CPPUNIT_ASSERT(df.id()==id); 
      {
	std::vector<edm::DataFrame::data_type> v2(10);
	std::copy(df.frame().begin(),df.frame().end(),v2.begin());
	CPPUNIT_ASSERT(sv==v2);
      }
      {
	std::vector<edm::DataFrame::data_type> v2(10);
	for (int i=0; i<10; i++){
	  EcalMGPASample s=df.sample(i);
	  v2[i]=s.raw();
	}
	CPPUNIT_ASSERT(sv==v2);
      }
    }
    int n;
    std::vector<edm::DataFrame::data_type> const & sv;
  };
}

template<typename DigiCollection>
void TestEcalDigi<DigiCollection>::iterator() {
  DigiCollection frames;
  for (int n=1;n<5;++n) {
    typename DigiCollection::DetId id = DigiCollection::DetId::unhashIndex(n);
    frames.push_back(id);
    typename DigiCollection::Digi df(frames.back());
    std::copy(sv.begin(),sv.end(),df.frame().begin());
  }
  // modern
  CPPUNIT_ASSERT(std::for_each(frames.begin(),frames.end(),
			       VerifyIter<typename DigiCollection::Digi>(this->sv)
			       ).n==4);
  // classical
  VerifyIter<typename DigiCollection::Digi> vi(sv);
  for (int n=0;n<int(frames.size());++n) {
    typename DigiCollection::Digi digi(frames[n]);
    vi(digi);
  } 
}



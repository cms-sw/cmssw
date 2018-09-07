#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"

#include<vector>
#include<algorithm>
#include<boost/function.hpp>

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
  
  void fill(int stride);
  //  void iter(int stride);


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


template<typename DigiCollection>
void TestEcalDigi<DigiCollection>::default_ctor() {

  DigiCollection frames;
  CPPUNIT_ASSERT(frames.stride()==10);
  // against enum in ExDetID
  CPPUNIT_ASSERT(frames.subdetId()==DigiCollection::DetId::Subdet);
  // against static metod in ExDetID
  CPPUNIT_ASSERT(frames.subdetId()==DigiCollection::DetId::subdet());
  DigiCollection smallframes(1);
  CPPUNIT_ASSERT(smallframes.stride()==1);

}

template<typename DigiCollection>
void TestEcalDigi<DigiCollection>::filling() {
  fill(10);
  fill(1);
}  

template<typename DigiCollection>
void TestEcalDigi<DigiCollection>::fill(int stride) {
  std::vector<edm::DataFrame::data_type> v1(stride);
  std::copy(sv.begin(),sv.begin()+stride,v1.begin());
  
  DigiCollection frames(stride);
  for (int n=1;n<5;++n) {
    typename DigiCollection::DetId id = DigiCollection::DetId::unhashIndex(n);
    frames.push_back(id);
    CPPUNIT_ASSERT(int(frames.size())==n);
    typename DigiCollection::Digi df(frames.back());
    CPPUNIT_ASSERT(df.size()==stride); 
    CPPUNIT_ASSERT(df.id()==id); 
    if (n%2==0) {
      // modern way of filling
      std::copy(sv.begin(),sv.begin()+stride,df.frame().begin());
      
      std::vector<edm::DataFrame::data_type> v2(stride);
      std::copy(frames.m_data.begin()+(n-1)*stride,frames.m_data.begin()+n*stride,v2.begin());
      CPPUNIT_ASSERT(v1==v2);
    } else {
      // classical way of filling
      for (int i=0; i<stride; i++)
	df.setSample(i,sv[i]);
      
      std::vector<edm::DataFrame::data_type> v2(stride);
      std::copy(frames.m_data.begin()+(n-1)*stride,frames.m_data.begin()+n*stride,v2.begin());
      CPPUNIT_ASSERT(v1==v2);
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

  void verifyBarrelId(edm::DataFrame::id_type id) {
    try {
      EBDetId detid{DetId(id)};
    } catch(...) {
      bool NotBarrelID=false;
      CPPUNIT_ASSERT(NotBarrelID);
    }
  }
  void verifyEndcapId(edm::DataFrame::id_type id) {
    try {
      EEDetId detid{DetId(id)}; // detid(id) does not throw
    } catch(...) {
      bool NotEndcapID=false;
      CPPUNIT_ASSERT(NotEndcapID);
    }
  }
  
  //one more way
  struct VerifyFrame {
    VerifyFrame(std::vector<edm::DataFrame::data_type> const & v): n(0), b(v), e(v){}
    
    void operator()(edm::DataFrame const & df) {
      DetId id(df.id());
      if (id.subdetId()==EcalBarrel) {
	b(df);n=b.n;
      } else if(id.subdetId()==EcalEndcap) {
	e(df);n=e.n;
      } else {
	bool WrongSubdetId=false;
	CPPUNIT_ASSERT(WrongSubdetId);
      }
    }
    int n;
    VerifyIter<EBDataFrame> b;
    VerifyIter<EEDataFrame> e;
  };


  // an alternative way
  void iterate(EcalDigiCollection const & frames) {
    boost::function<void(edm::DataFrame::id_type)> verifyId;
    if (frames.subdetId()==EcalBarrel)
      verifyId=verifyBarrelId;
    else if(frames.subdetId()==EcalEndcap)
      verifyId=verifyEndcapId;
    else {
      bool WrongSubdetId=false;
      CPPUNIT_ASSERT(WrongSubdetId);
    }

    //    std::for_each(frames.begin(),frames.end(),
    //		  boost::bind(verifyId,boost::bind(&edm::DataFrame::id,_1)));
    // same as above....
    for (int n=0;n<int(frames.size());++n) {
      edm::DataFrame df = frames[n];
      verifyId(df.id());
    }

  }

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
  // alternatives
  iterate(frames);
  CPPUNIT_ASSERT(std::for_each(frames.begin(),frames.end(),VerifyFrame(sv)).n==4);
}



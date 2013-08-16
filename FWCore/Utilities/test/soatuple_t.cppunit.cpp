/*----------------------------------------------------------------------

Test program for edm::SoATuple class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/SoATuple.h"

class testSoATuple: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testSoATuple);
  
  CPPUNIT_TEST(builtinTest);
  CPPUNIT_TEST(classTest);
  CPPUNIT_TEST(badPaddingTest);
  CPPUNIT_TEST(growthTest);
  CPPUNIT_TEST(loopTest);
  CPPUNIT_TEST(copyConstructorTest);
  CPPUNIT_TEST(assignmentTest);
  CPPUNIT_TEST(emplace_backTest);
  CPPUNIT_TEST(alignmentTest);
  
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void builtinTest();
  void badPaddingTest();
  void classTest();
  void growthTest();
  void loopTest();
  void copyConstructorTest();
  void assignmentTest();
  void emplace_backTest();
  void alignmentTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSoATuple);

void testSoATuple::builtinTest()
{
  edm::SoATuple<int,float,bool> s;
  
  s.reserve(3);
  CPPUNIT_ASSERT(s.size() == 0);
  
  s.push_back(std::make_tuple(int{1},float{3.2},false));
  //std::cout <<s.get<1>(0)<<std::endl;
  CPPUNIT_ASSERT(s.size() == 1);
  CPPUNIT_ASSERT(1==s.get<0>(0));
  CPPUNIT_ASSERT(float{3.2}==s.get<1>(0));
  CPPUNIT_ASSERT(false==s.get<2>(0));
    
  s.push_back(std::make_tuple(int{2},float{3.1415},true));
  CPPUNIT_ASSERT(s.size() == 2);
  CPPUNIT_ASSERT(1==s.get<0>(0));
  CPPUNIT_ASSERT(float{3.2}==s.get<1>(0));
  CPPUNIT_ASSERT(false==s.get<2>(0));
  CPPUNIT_ASSERT(2==s.get<0>(1));
  CPPUNIT_ASSERT(float{3.1415}==s.get<1>(1));
  CPPUNIT_ASSERT(true==s.get<2>(1));
  
  s.push_back(std::make_tuple(int{-1},float{58.6},true));
  CPPUNIT_ASSERT(s.size() == 3);
  CPPUNIT_ASSERT(1==s.get<0>(0));
  CPPUNIT_ASSERT(float{3.2}==s.get<1>(0));
  CPPUNIT_ASSERT(false==s.get<2>(0));
  CPPUNIT_ASSERT(2==s.get<0>(1));
  CPPUNIT_ASSERT(float{3.1415}==s.get<1>(1));
  CPPUNIT_ASSERT(true==s.get<2>(1));
  CPPUNIT_ASSERT(-1==s.get<0>(2));
  CPPUNIT_ASSERT(float{58.6}==s.get<1>(2));
  CPPUNIT_ASSERT(true==s.get<2>(2));
}

void testSoATuple::badPaddingTest()
{
  edm::SoATuple<bool,int,double> s;
  
  s.reserve(3);
  CPPUNIT_ASSERT(s.size() == 0);
  
  s.push_back(std::make_tuple(false,int{1},double{3.2}));
  //std::cout <<s.get<1>(0)<<std::endl;
  CPPUNIT_ASSERT(s.size() == 1);
  CPPUNIT_ASSERT(1==s.get<1>(0));
  CPPUNIT_ASSERT(double{3.2}==s.get<2>(0));
  CPPUNIT_ASSERT(false==s.get<0>(0));
  
  s.push_back(std::make_tuple(true,int{2},double{3.1415}));
  CPPUNIT_ASSERT(s.size() == 2);
  CPPUNIT_ASSERT(1==s.get<1>(0));
  CPPUNIT_ASSERT(double{3.2}==s.get<2>(0));
  CPPUNIT_ASSERT(false==s.get<0>(0));
  CPPUNIT_ASSERT(2==s.get<1>(1));
  CPPUNIT_ASSERT(double{3.1415}==s.get<2>(1));
  CPPUNIT_ASSERT(true==s.get<0>(1));
  
  s.push_back(std::make_tuple(true,int{-1},double{58.6}));
  CPPUNIT_ASSERT(s.size() == 3);
  CPPUNIT_ASSERT(1==s.get<1>(0));
  CPPUNIT_ASSERT(double{3.2}==s.get<2>(0));
  CPPUNIT_ASSERT(false==s.get<0>(0));
  CPPUNIT_ASSERT(2==s.get<1>(1));
  CPPUNIT_ASSERT(double{3.1415}==s.get<2>(1));
  CPPUNIT_ASSERT(true==s.get<0>(1));
  CPPUNIT_ASSERT(-1==s.get<1>(2));
  CPPUNIT_ASSERT(double{58.6}==s.get<2>(2));
  CPPUNIT_ASSERT(true==s.get<0>(2));
}

namespace {
  struct ConstructDestructCounter {
    static unsigned int s_constructorCalls;
    static unsigned int s_destructorCalls;
    
    //make sure the destructor is being called on the correct memory location
    void* originalThis;

    ConstructDestructCounter(): originalThis(this) {
      ++s_constructorCalls;
    }

    ConstructDestructCounter(ConstructDestructCounter const&):originalThis(this) {
      ++s_constructorCalls;
    }

    ConstructDestructCounter(ConstructDestructCounter&&):originalThis(this) {
      ++s_constructorCalls;
    }
    
    ~ConstructDestructCounter() {
      CPPUNIT_ASSERT(originalThis==this);
      ++s_destructorCalls;
    }
    
  };
  unsigned int ConstructDestructCounter::s_constructorCalls = 0;
  unsigned int ConstructDestructCounter::s_destructorCalls = 0;
}


void testSoATuple::classTest()
{
  ConstructDestructCounter::s_constructorCalls = 0;
  ConstructDestructCounter::s_destructorCalls = 0;
  CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls);
  {
    edm::SoATuple<std::string, ConstructDestructCounter> s;
    
    s.reserve(3);
    const std::string kFoo{"foo"};
    CPPUNIT_ASSERT(s.size() == 0);
    {
      ConstructDestructCounter dummy;
      s.push_back(std::make_tuple(kFoo,dummy));
    }
    CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls+1);
    CPPUNIT_ASSERT(s.size()==1);
    CPPUNIT_ASSERT(kFoo==s.get<0>(0));

    const std::string kBar{"bar"};
    {
      ConstructDestructCounter dummy;
      s.push_back(std::make_tuple(kBar,dummy));
    }
    CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls+2);
    CPPUNIT_ASSERT(s.size()==2);
    CPPUNIT_ASSERT(kFoo==s.get<0>(0));
    CPPUNIT_ASSERT(kBar==s.get<0>(1));
  }
  CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls);
}

void testSoATuple::growthTest()
{
  ConstructDestructCounter::s_constructorCalls = 0;
  ConstructDestructCounter::s_destructorCalls = 0;
  CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls);
  {
    edm::SoATuple<unsigned int, ConstructDestructCounter> s;
    CPPUNIT_ASSERT(s.size() == 0);
    CPPUNIT_ASSERT(s.capacity() == 0);
    for(unsigned int i= 0; i< 100; ++i) {
      {
        ConstructDestructCounter dummy;
        s.push_back(std::make_tuple(i,dummy));
      }
      CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls+i+1);
      CPPUNIT_ASSERT(s.size()==i+1);
      CPPUNIT_ASSERT(s.capacity()>=i+1);
      for(unsigned int j=0; j<s.size(); ++j) {
        CPPUNIT_ASSERT(j==s.get<0>(j));
      }
    }
    
    s.shrink_to_fit();
    CPPUNIT_ASSERT(s.capacity()==s.size());
    CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls-ConstructDestructCounter::s_destructorCalls == s.size());
    for(unsigned int j=0; j<s.size(); ++j) {
      CPPUNIT_ASSERT(j==s.get<0>(j));
    }

  }
  
  CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls);
}

void testSoATuple::copyConstructorTest()
{
  ConstructDestructCounter::s_constructorCalls = 0;
  ConstructDestructCounter::s_destructorCalls = 0;
  CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls);
  {
    edm::SoATuple<unsigned int, ConstructDestructCounter> s;
    CPPUNIT_ASSERT(s.size() == 0);
    CPPUNIT_ASSERT(s.capacity() == 0);
    for(unsigned int i= 0; i< 100; ++i) {
      {
        ConstructDestructCounter dummy;
        s.push_back(std::make_tuple(i,dummy));
      }
      edm::SoATuple<unsigned int, ConstructDestructCounter> sCopy(s);
      CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls+2*i+2);
      CPPUNIT_ASSERT(s.size()==i+1);
      CPPUNIT_ASSERT(s.capacity()>=i+1);
      CPPUNIT_ASSERT(sCopy.size()==i+1);
      CPPUNIT_ASSERT(sCopy.capacity()>=i+1);

      for(unsigned int j=0; j<s.size(); ++j) {
        CPPUNIT_ASSERT(j==s.get<0>(j));
        CPPUNIT_ASSERT(j==sCopy.get<0>(j));
      }
    }
  }
  CPPUNIT_ASSERT(ConstructDestructCounter::s_constructorCalls==ConstructDestructCounter::s_destructorCalls);
}

void testSoATuple::assignmentTest()
{
  const std::vector<std::string> sValues = {"foo","fii","fee"};
  edm::SoATuple<std::string, int> s;
  s.reserve(sValues.size());
  int i = 0;
  for(auto const& v: sValues) {
    s.push_back(std::make_tuple(v,i));
    ++i;
  }
  
  edm::SoATuple<std::string,int> sAssign;
  sAssign.reserve(2);
  sAssign.push_back(std::make_tuple("barney",10));
  sAssign.push_back(std::make_tuple("fred",7));
  CPPUNIT_ASSERT(sAssign.size()==2);
  
  sAssign = s;
  CPPUNIT_ASSERT(sAssign.size()==s.size());
  CPPUNIT_ASSERT(sAssign.size()==sValues.size());

  i = 0;
  for(auto const& v: sValues) {
    CPPUNIT_ASSERT(v == sAssign.get<0>(i));
    CPPUNIT_ASSERT(i == sAssign.get<1>(i));
    ++i;
  }
}


void testSoATuple::loopTest()
{
  edm::SoATuple<int,int,int> s;
  s.reserve(50);
  for(int i=0; i<50;++i) {
    s.push_back(std::make_tuple(i,i+1,i+2));
  }
  CPPUNIT_ASSERT(50==s.size());
  int index=0;
  for(auto it =s.begin<0>(), itEnd = s.end<0>(); it!=itEnd;++it,++index) {
    CPPUNIT_ASSERT(index == *it);
  }
  index = 1;
  for(auto it =s.begin<1>(), itEnd = s.end<1>(); it!=itEnd;++it,++index) {
    CPPUNIT_ASSERT(index == *it);
  }

  index = 2;
  for(auto it =s.begin<2>(), itEnd = s.end<2>(); it!=itEnd;++it,++index) {
    CPPUNIT_ASSERT(index == *it);
  }
}

void testSoATuple::emplace_backTest()
{
  const std::vector<std::string> sValues = {"foo","fii","fee"};
  edm::SoATuple<std::string, int> s;
  s.reserve(sValues.size());
  int i = 0;
  for(auto const& v: sValues) {
    s.emplace_back(v,i);
    ++i;
  }
  i = 0;
  for(auto const& v: sValues) {
    CPPUNIT_ASSERT(v == s.get<0>(i));
    CPPUNIT_ASSERT(i == s.get<1>(i));
    ++i;
  }
}

namespace  {
  struct CharDummy {
    char i;
    char j;
  };
  
  struct ComplexDummy {
    char * p;
    double f;
  };
}
void testSoATuple::alignmentTest()
{
  CPPUNIT_ASSERT((alignof(double)==edm::soahelper::SoATupleHelper<2,double,bool>::max_alignment));
  CPPUNIT_ASSERT((alignof(double)==edm::soahelper::SoATupleHelper<2,bool,double>::max_alignment));
  CPPUNIT_ASSERT((alignof(float)==edm::soahelper::SoATupleHelper<2,float,bool>::max_alignment));
  CPPUNIT_ASSERT((alignof(float)==edm::soahelper::SoATupleHelper<2,bool,float>::max_alignment));
  //std::cout <<"alignment of CharDummy "<<alignof(CharDummy)<<" alignment of char "<<alignof(char)<<std::endl;
  CPPUNIT_ASSERT((alignof(CharDummy)==edm::soahelper::SoATupleHelper<2,char,CharDummy>::max_alignment));
  CPPUNIT_ASSERT((alignof(CharDummy)==edm::soahelper::SoATupleHelper<2,CharDummy,char>::max_alignment));
  CPPUNIT_ASSERT((alignof(ComplexDummy)==edm::soahelper::SoATupleHelper<2,char,ComplexDummy>::max_alignment));
  CPPUNIT_ASSERT((alignof(ComplexDummy)==edm::soahelper::SoATupleHelper<2,ComplexDummy,char>::max_alignment));
  
  CPPUNIT_ASSERT((alignof(float)==edm::soahelper::SoATupleHelper<2,float,float>::max_alignment));
  CPPUNIT_ASSERT((16==edm::soahelper::SoATupleHelper<2,edm::AlignedVec<float>,edm::AlignedVec<float>>::max_alignment));

  CPPUNIT_ASSERT((alignof(double)==edm::soahelper::SoATupleHelper<2,double,double>::max_alignment));
  CPPUNIT_ASSERT((16==edm::soahelper::SoATupleHelper<2,edm::AlignedVec<double>,edm::AlignedVec<double>>::max_alignment));

  edm::SoATuple<edm::AlignedVec<float>,edm::AlignedVec<float>,edm::AlignedVec<float>> vFloats;
  vFloats.reserve(50);
  for(unsigned int i=0; i<50;++i) {
    vFloats.emplace_back(1.0f,2.0f,3.0f);
  }
  CPPUNIT_ASSERT(reinterpret_cast<std::intptr_t>(vFloats.begin<0>()) %16 == 0);
  CPPUNIT_ASSERT(reinterpret_cast<std::intptr_t>(vFloats.begin<1>()) %16 == 0);
  CPPUNIT_ASSERT(reinterpret_cast<std::intptr_t>(vFloats.begin<2>()) %16 == 0);
}

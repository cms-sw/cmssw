// -*- C++ -*-
//
// Package:     Histograms
// Class  :     metoemdformat_t
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Tue Jul 21 11:00:06 CDT 2009
// $Id: metoedmformat_t.cc,v 1.1 2009/07/21 17:53:30 chrjones Exp $
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

// user include files
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

class TestMEtoEDMFormat : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestMEtoEDMFormat);
  CPPUNIT_TEST(testMergeInt);
  CPPUNIT_TEST(testMergeDouble);
  CPPUNIT_TEST(testMergeTString);
  CPPUNIT_TEST(testMergeT);
  CPPUNIT_TEST_SUITE_END();
public:
  void testMergeInt();
  void testMergeDouble();
  void testMergeTString();
  void testMergeT();
  void setUp() {}
  void tearDown() {}
};


///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestMEtoEDMFormat);

template <typename T>
bool areEquivalent(const MEtoEDM<T>& iLHS, const MEtoEDM<T>& iRHS) {
  if(iLHS.getMEtoEdmObject().size() != iRHS.getMEtoEdmObject().size()) {
    std::cout <<"areEquivalent failure: sizes are different"<<std::endl;
    return false;
  }
  typedef typename MEtoEDM<T>::MEtoEdmObjectVector Vec;
  typename Vec::const_iterator itRHS = iRHS.getMEtoEdmObject().begin();
  for(typename Vec::const_iterator itLHS = iLHS.getMEtoEdmObject().begin(), itLHSEnd = iLHS.getMEtoEdmObject().end();
      itLHS != itLHSEnd;
      ++itLHS,++itRHS) {
    if( itLHS->name != itRHS->name ||
	itLHS->tags != itRHS->tags ||
	itLHS->object != itRHS->object ||
	itLHS->release != itRHS->release ||
	itLHS->run != itRHS->run ||
	itLHS->datatier != itRHS->datatier) {
      std::cout <<"areEquivalent failure: elements '"<<(itLHS->name)<<"' "<<(itLHS->object)
		<<" and '"<<(itRHS->name)<<"' "<<(itRHS->object)<<" are different"<<std::endl;
      return false;
    }
  }
  return true;
}

void
TestMEtoEDMFormat::testMergeInt()
{
  MEtoEDM<int> empty(0);

  MEtoEDM<int>::TagList tList;
  MEtoEDM<int> full(3);
  full.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<int> fullReordered(3);
  fullReordered.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<int> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  MEtoEDM<int> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2,full));

  MEtoEDM<int> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3,full));
  
  MEtoEDM<int> part1(2);
  part1.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  part1.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  MEtoEDM<int> part2(1);
  part2.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1,full));

  MEtoEDM<int> specials1(3);
  specials1.putMEtoEdmObject("EventInfo/processedEvents",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  specials1.putMEtoEdmObject("EventInfo/iEvent",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  specials1.putMEtoEdmObject("EventInfo/iLumiSection",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<int> specials2(3);
  specials2.putMEtoEdmObject("EventInfo/processedEvents",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  specials2.putMEtoEdmObject("EventInfo/iEvent",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");
  specials2.putMEtoEdmObject("EventInfo/iLumiSection",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<int> specialsTotal(3);
  specialsTotal.putMEtoEdmObject("EventInfo/processedEvents",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  specialsTotal.putMEtoEdmObject("EventInfo/iEvent",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");
  specialsTotal.putMEtoEdmObject("EventInfo/iLumiSection",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");
  specials1.mergeProduct(specials2);
  CPPUNIT_ASSERT(areEquivalent(specials1,specialsTotal));
}


void
TestMEtoEDMFormat::testMergeDouble()
{
  MEtoEDM<double> empty(0);

  MEtoEDM<double>::TagList tList;
  MEtoEDM<double> full(3);
  full.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<double> fullReordered(3);
  fullReordered.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<double> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  MEtoEDM<double> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2,full));

  MEtoEDM<double> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3,full));
  
  MEtoEDM<double> part1(2);
  part1.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  part1.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  MEtoEDM<double> part2(1);
  part2.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1,full));

}

/*
//NOTE: TString doesn't have a standard comparision operator
bool operator!=(const TString& iLHS, const TString& iRHS) {
  return iLHS.CompareTo(iRHS);
}
*/

void
TestMEtoEDMFormat::testMergeTString()
{
  MEtoEDM<TString> empty(0);

  MEtoEDM<TString>::TagList tList;
  MEtoEDM<TString> full(3);
  full.putMEtoEdmObject("a",
			tList,
			"1",
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("b",
			tList,
			"2",
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("c",
			tList,
			"3",
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<TString> fullReordered(3);
  fullReordered.putMEtoEdmObject("b",
			tList,
			"2",
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("a",
			tList,
			"1",
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("c",
			tList,
			"3",
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<TString> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  MEtoEDM<TString> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2,full));

  MEtoEDM<TString> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3,full));
  
  MEtoEDM<TString> part1(2);
  part1.putMEtoEdmObject("a",
			tList,
			"1",
			"TEST",
			12,
			"MYTIER");
  part1.putMEtoEdmObject("b",
			tList,
			"2",
			"TEST",
			12,
			"MYTIER");
  MEtoEDM<TString> part2(1);
  part2.putMEtoEdmObject("c",
			tList,
			"3",
			"TEST",
			12,
			"MYTIER");
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1,full));

}

namespace {
  struct Axis {
    Axis() {}
    float GetXmin() const { return 0;}
    float GetXmax() const {return 1;}
    
    static const Axis dummy;
  };

  const Axis Axis::dummy;

  struct Dummy {
    Dummy(int i) : m_i(i) {}
    Dummy() : m_i(0) {}
    void Add(const Dummy* iOther) { m_i += iOther->m_i; 
    //std::cout <<"Dummy Add "<<m_i<<" "<<iOther->m_i<<std::endl;
    }
    
    bool operator!=( const Dummy& iOther) const {
      return m_i != iOther.m_i;
    }

    int GetNbinsX() const { return 1;}
    int GetNbinsY() const { return 1;}
    int GetNbinsZ() const { return 1;}

    const Axis* GetXaxis() const {return &Axis::dummy;}
    const Axis* GetYaxis() const {return &Axis::dummy;}
    const Axis* GetZaxis() const {return &Axis::dummy;}

    int m_i;

  };

    std::ostream& operator<<(std::ostream& os, const Dummy& iDummy) {
      os<<iDummy.m_i;
      return os;
    }

}


void
TestMEtoEDMFormat::testMergeT()
{
  MEtoEDM<Dummy> empty(0);

  MEtoEDM<Dummy>::TagList tList;
  MEtoEDM<Dummy> full(3);
  full.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  full.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<Dummy> fullReordered(3);
  fullReordered.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  fullReordered.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");

  MEtoEDM<Dummy> doubleFull(3);
  doubleFull.putMEtoEdmObject("a",
			      tList,
			      2*1,
			      "TEST",
			      12,
			      "MYTIER");
  doubleFull.putMEtoEdmObject("b",
			      tList,
			      2*2,
			      "TEST",
			      12,
			      "MYTIER");
  doubleFull.putMEtoEdmObject("c",
			      tList,
			      2*3,
			      "TEST",
			      12,
			      "MYTIER");

  MEtoEDM<Dummy> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge,full));

  MEtoEDM<Dummy> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2,full));

  MEtoEDM<Dummy> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3,doubleFull));
  
  MEtoEDM<Dummy> part1(2);
  part1.putMEtoEdmObject("a",
			tList,
			1,
			"TEST",
			12,
			"MYTIER");
  part1.putMEtoEdmObject("b",
			tList,
			2,
			"TEST",
			12,
			"MYTIER");
  MEtoEDM<Dummy> part2(1);
  part2.putMEtoEdmObject("c",
			tList,
			3,
			"TEST",
			12,
			"MYTIER");
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1,full));

}

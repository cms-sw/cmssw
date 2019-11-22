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
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

// user include files
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "TAxis.h"

class TestMEtoEDMFormat : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestMEtoEDMFormat);
  CPPUNIT_TEST(testMergeInt64);
  CPPUNIT_TEST(testMergeDouble);
  CPPUNIT_TEST(testMergeTString);
  CPPUNIT_TEST(testMergeT);
  CPPUNIT_TEST_SUITE_END();

public:
  void testMergeInt64();
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
  if (iLHS.getMEtoEdmObject().size() != iRHS.getMEtoEdmObject().size()) {
    std::cout << "areEquivalent failure: sizes are different" << std::endl;
    return false;
  }
  typedef typename MEtoEDM<T>::MEtoEdmObjectVector Vec;
  typename Vec::const_iterator itRHS = iRHS.getMEtoEdmObject().begin();
  for (typename Vec::const_iterator itLHS = iLHS.getMEtoEdmObject().begin(), itLHSEnd = iLHS.getMEtoEdmObject().end();
       itLHS != itLHSEnd;
       ++itLHS, ++itRHS) {
    if (itLHS->name != itRHS->name || itLHS->tags != itRHS->tags || itLHS->object != itRHS->object) {
      std::cout << "areEquivalent failure: elements '" << (itLHS->name) << "' " << (itLHS->object) << " and '"
                << (itRHS->name) << "' " << (itRHS->object) << " are different" << std::endl;
      return false;
    }
  }
  return true;
}

void TestMEtoEDMFormat::testMergeInt64() {
  MEtoEDM<long long> empty(0);

  MEtoEDM<long long> full(3);
  full.putMEtoEdmObject("a", 1);
  full.putMEtoEdmObject("b", 2);
  full.putMEtoEdmObject("c", 3);

  MEtoEDM<long long> fullReordered(3);
  fullReordered.putMEtoEdmObject("b", 2);
  fullReordered.putMEtoEdmObject("a", 1);
  fullReordered.putMEtoEdmObject("c", 3);

  MEtoEDM<long long> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  MEtoEDM<long long> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2, full));

  MEtoEDM<long long> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3, full));

  MEtoEDM<long long> part1(2);
  part1.putMEtoEdmObject("a", 1);
  part1.putMEtoEdmObject("b", 2);
  MEtoEDM<long long> part2(1);
  part2.putMEtoEdmObject("c", 3);
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1, full));

  MEtoEDM<long long> specials1(3);
  specials1.putMEtoEdmObject("EventInfo/processedEvents", 1);
  specials1.putMEtoEdmObject("EventInfo/iEvent", 2);
  specials1.putMEtoEdmObject("EventInfo/iLumiSection", 3);

  MEtoEDM<long long> specials2(3);
  specials2.putMEtoEdmObject("EventInfo/processedEvents", 1);
  specials2.putMEtoEdmObject("EventInfo/iEvent", 3);
  specials2.putMEtoEdmObject("EventInfo/iLumiSection", 2);

  MEtoEDM<long long> specialsTotal(3);
  specialsTotal.putMEtoEdmObject("EventInfo/processedEvents", 2);
  specialsTotal.putMEtoEdmObject("EventInfo/iEvent", 3);
  specialsTotal.putMEtoEdmObject("EventInfo/iLumiSection", 3);
  specials1.mergeProduct(specials2);
  CPPUNIT_ASSERT(areEquivalent(specials1, specialsTotal));
}

void TestMEtoEDMFormat::testMergeDouble() {
  MEtoEDM<double> empty(0);

  MEtoEDM<double> full(3);
  full.putMEtoEdmObject("a", 1);
  full.putMEtoEdmObject("b", 2);
  full.putMEtoEdmObject("c", 3);

  MEtoEDM<double> fullReordered(3);
  fullReordered.putMEtoEdmObject("b", 2);
  fullReordered.putMEtoEdmObject("a", 1);
  fullReordered.putMEtoEdmObject("c", 3);

  MEtoEDM<double> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  MEtoEDM<double> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2, full));

  MEtoEDM<double> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3, full));

  MEtoEDM<double> part1(2);
  part1.putMEtoEdmObject("a", 1);
  part1.putMEtoEdmObject("b", 2);
  MEtoEDM<double> part2(1);
  part2.putMEtoEdmObject("c", 3);
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1, full));
}

/*
//NOTE: TString doesn't have a standard comparision operator
bool operator!=(const TString& iLHS, const TString& iRHS) {
  return iLHS.CompareTo(iRHS);
}
*/

void TestMEtoEDMFormat::testMergeTString() {
  MEtoEDM<TString> empty(0);

  MEtoEDM<TString> full(3);
  full.putMEtoEdmObject("a", "1");
  full.putMEtoEdmObject("b", "2");
  full.putMEtoEdmObject("c", "3");

  MEtoEDM<TString> fullReordered(3);
  fullReordered.putMEtoEdmObject("b", "2");
  fullReordered.putMEtoEdmObject("a", "1");
  fullReordered.putMEtoEdmObject("c", "3");

  MEtoEDM<TString> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  MEtoEDM<TString> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2, full));

  MEtoEDM<TString> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3, full));

  MEtoEDM<TString> part1(2);
  part1.putMEtoEdmObject("a", "1");
  part1.putMEtoEdmObject("b", "2");
  MEtoEDM<TString> part2(1);
  part2.putMEtoEdmObject("c", "3");
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1, full));
}

namespace {
  const TAxis dummy(1.0, 0.0, 1.0);

  struct Dummy {
    Dummy(int i) : m_i(i) {}
    Dummy() : m_i(0) {}
    void Add(const Dummy* iOther) {
      m_i += iOther->m_i;
      //std::cout <<"Dummy Add "<<m_i<<" "<<iOther->m_i<<std::endl;
    }

    bool operator!=(const Dummy& iOther) const { return m_i != iOther.m_i; }

    int GetNbinsX() const { return 1; }
    int GetNbinsY() const { return 1; }
    int GetNbinsZ() const { return 1; }

    const TAxis* GetXaxis() const { return &dummy; }
    const TAxis* GetYaxis() const { return &dummy; }
    const TAxis* GetZaxis() const { return &dummy; }

    bool CanExtendAllAxes() const { return false; }
    long long Merge(void*) { return -1; }

    int m_i;
  };

  std::ostream& operator<<(std::ostream& os, const Dummy& iDummy) {
    os << iDummy.m_i;
    return os;
  }

}  // namespace

void TestMEtoEDMFormat::testMergeT() {
  MEtoEDM<Dummy> empty(0);

  MEtoEDM<Dummy> full(3);
  full.putMEtoEdmObject("a", 1);
  full.putMEtoEdmObject("b", 2);
  full.putMEtoEdmObject("c", 3);

  MEtoEDM<Dummy> fullReordered(3);
  fullReordered.putMEtoEdmObject("b", 2);
  fullReordered.putMEtoEdmObject("a", 1);
  fullReordered.putMEtoEdmObject("c", 3);

  MEtoEDM<Dummy> doubleFull(3);
  doubleFull.putMEtoEdmObject("a", 2 * 1);
  doubleFull.putMEtoEdmObject("b", 2 * 2);
  doubleFull.putMEtoEdmObject("c", 2 * 3);

  MEtoEDM<Dummy> toMerge(full);

  //be sure copy constructor worked
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  toMerge.mergeProduct(empty);
  CPPUNIT_ASSERT(areEquivalent(toMerge, full));

  MEtoEDM<Dummy> toMerge2(empty);
  toMerge2.mergeProduct(full);
  CPPUNIT_ASSERT(areEquivalent(toMerge2, full));

  MEtoEDM<Dummy> toMerge3(full);
  toMerge3.mergeProduct(fullReordered);
  CPPUNIT_ASSERT(areEquivalent(toMerge3, doubleFull));

  MEtoEDM<Dummy> part1(2);
  part1.putMEtoEdmObject("a", 1);
  part1.putMEtoEdmObject("b", 2);
  MEtoEDM<Dummy> part2(1);
  part2.putMEtoEdmObject("c", 3);
  part1.mergeProduct(part2);
  CPPUNIT_ASSERT(areEquivalent(part1, full));
}

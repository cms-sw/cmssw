#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/CopyPolicy.h"



class testRangeMap : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testRangeMap);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testRangeMap);

#include <iostream>

struct MatchOddId {
  bool operator()(int i) const { return i % 2 == 1; }
};


class IntComparator {
public:
  bool operator()(int d1, int d2) const {
    //
    // stupid, 3 and 2 are treated as the same thing
    //
    if ((d1 == 4 && d2 ==3) ||
	(d1 == 3 && d2 ==4)) return false;
    return d1 < d2;
  }
};

void testRangeMap::checkAll() {
  typedef edm::RangeMap<int, std::vector<int>, edm::CopyPolicy<int> > map;

  map m;
  int v3[] = { 1, 2, 3, 4 };
  int v4[] = { 10, 11, 12};
  int v2[] = { 5, 6, 7 };
  int v1[] = { 8, 9 };
  int s1 = sizeof(v1)/sizeof(int);
  int s2 = sizeof(v2)/sizeof(int);
  int s3 = sizeof(v3)/sizeof(int);
  int s4 = sizeof(v4)/sizeof(int);
  m.put(3, v3, v3 + s3);
  m.put(2, v2, v2 + s2);
  m.put(1, v1, v1 + s1);
  m.put(4, v4, v4 + s4);

  m.post_insert();

  map::const_iterator i = m.begin();
  CPPUNIT_ASSERT(*i++ == 8);
  CPPUNIT_ASSERT(*i++ == 9);
  CPPUNIT_ASSERT(*i++ == 5);
  CPPUNIT_ASSERT(*i++ == 6);
  CPPUNIT_ASSERT(*i++ == 7);
  CPPUNIT_ASSERT(*i++ == 1);
  CPPUNIT_ASSERT(*i++ == 2);
  CPPUNIT_ASSERT(*i++ == 3);
  CPPUNIT_ASSERT(*i++ == 4);
  CPPUNIT_ASSERT(*i++ == 10);
  CPPUNIT_ASSERT(*i++ == 11);
  CPPUNIT_ASSERT(*i++ == 12);
  CPPUNIT_ASSERT(i == m.end());


  CPPUNIT_ASSERT(* --i == 12);
  CPPUNIT_ASSERT(* --i == 11);
  CPPUNIT_ASSERT(* --i == 10);
  CPPUNIT_ASSERT(* --i == 4);
  CPPUNIT_ASSERT(* --i == 3);
  CPPUNIT_ASSERT(* --i == 2);
  CPPUNIT_ASSERT(* --i == 1);
  CPPUNIT_ASSERT(* --i == 7);
  CPPUNIT_ASSERT(* --i == 6);
  CPPUNIT_ASSERT(* --i == 5);
  CPPUNIT_ASSERT(* --i == 9);
  CPPUNIT_ASSERT(* --i == 8);
  CPPUNIT_ASSERT(i == m.begin());


  map::id_iterator b = m.id_begin();
  CPPUNIT_ASSERT(m.id_size() == 4);
  CPPUNIT_ASSERT(*b++ == 1);
  CPPUNIT_ASSERT(*b++ == 2);
  CPPUNIT_ASSERT(*b++ == 3);
  CPPUNIT_ASSERT(*b++ == 4);
  CPPUNIT_ASSERT(b == m.id_end());

  CPPUNIT_ASSERT(* --b == 4);
  CPPUNIT_ASSERT(* --b == 3);
  CPPUNIT_ASSERT(* --b == 2);
  CPPUNIT_ASSERT(* --b == 1);
  CPPUNIT_ASSERT(b == m.id_begin());
  //
  // try the get
  //
  map::range r = m.get(1);
  CPPUNIT_ASSERT(r.second - r.first == 2);
  r = m.get(2);
  CPPUNIT_ASSERT(r.second - r.first == 3);
  r = m.get(3);
  CPPUNIT_ASSERT(r.second - r.first == 4);
  r = m.get(4);
  CPPUNIT_ASSERT(r.second - r.first == 3);

  //
  // try the get with comparator
  //
  r = m.get(3,IntComparator());
  CPPUNIT_ASSERT(r.second - r.first == 7);
  i = r.first;

  CPPUNIT_ASSERT(*i++ == 1);
  CPPUNIT_ASSERT(*i++ == 2);
  CPPUNIT_ASSERT(*i++ == 3);
  CPPUNIT_ASSERT(*i++ == 4);
  CPPUNIT_ASSERT(*i++ == 10);
  CPPUNIT_ASSERT(*i++ == 11);
  CPPUNIT_ASSERT(*i++ == 12);
  CPPUNIT_ASSERT(i == r.second);


  r = m.get(1,IntComparator());
  CPPUNIT_ASSERT(r.second - r.first == 2);

  }


#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/IDVectorMap.h"
#include "DataFormats/Common/interface/CopyPolicy.h"

class testIDVectorMap : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testIDVectorMap);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testIDVectorMap);

#include <iostream>

struct MatchOddId {
  bool operator()(int i) const { return i % 2 == 1; }
};

void testIDVectorMap::checkAll() {
  typedef edm::IDVectorMap<int, std::vector<int>, edm::CopyPolicy<int> > map;
  map m;
  int v1[] = { 1, 2, 3, 4 };
  int v2[] = { 5, 6, 7 };
  int v3[] = { 8, 9 };
  int s1 = sizeof(v1)/sizeof(int);
  int s2 = sizeof(v2)/sizeof(int);
  int s3 = sizeof(v3)/sizeof(int);
  for(int i = 0; i < s1; ++i) m.insert(1, v1[i]);
  //  for(int i = 0; i < s2; ++i) m.insert(2, v2[i]);
  m.insert(2, v2, v2 + s2);
  for(int i = 0; i < s3; ++i) m.insert(3, v3[i]);

  map::const_iterator i = m.begin();
  CPPUNIT_ASSERT(*i++ == 1);
  CPPUNIT_ASSERT(*i++ == 2);
  CPPUNIT_ASSERT(*i++ == 3);
  CPPUNIT_ASSERT(*i++ == 4);
  CPPUNIT_ASSERT(*i++ == 5);
  CPPUNIT_ASSERT(*i++ == 6);
  CPPUNIT_ASSERT(*i++ == 7);
  CPPUNIT_ASSERT(*i++ == 8);
  CPPUNIT_ASSERT(*i++ == 9);
  CPPUNIT_ASSERT(i == m.end());

  CPPUNIT_ASSERT(* --i == 9);
  CPPUNIT_ASSERT(* --i == 8);
  CPPUNIT_ASSERT(* --i == 7);
  CPPUNIT_ASSERT(* --i == 6);
  CPPUNIT_ASSERT(* --i == 5);
  CPPUNIT_ASSERT(* --i == 4);
  CPPUNIT_ASSERT(* --i == 3);
  CPPUNIT_ASSERT(* --i == 2);
  CPPUNIT_ASSERT(* --i == 1);
  CPPUNIT_ASSERT(i == m.begin());


  map::id_iterator b = m.id_begin();
  CPPUNIT_ASSERT(m.id_size() == 3);
  CPPUNIT_ASSERT(* b ++ == 1);
  CPPUNIT_ASSERT(* b ++ == 2);
  CPPUNIT_ASSERT(* b ++ == 3);
  CPPUNIT_ASSERT(b == m.id_end());

  CPPUNIT_ASSERT(* -- b == 3);
  CPPUNIT_ASSERT(* -- b == 2);
  CPPUNIT_ASSERT(* -- b == 1);
  CPPUNIT_ASSERT(b == m.id_begin());

  map::range r = m.get(1);
  CPPUNIT_ASSERT(r.end - r.begin == s1);
  map::container_iterator j = r.begin;
  CPPUNIT_ASSERT(* j ++ ==  1);
  CPPUNIT_ASSERT(* j ++ ==  2);
  CPPUNIT_ASSERT(* j ++ ==  3);
  CPPUNIT_ASSERT(* j ++ ==  4);
  CPPUNIT_ASSERT(j == r.end);

  MatchOddId o;
  map::match_iterator<MatchOddId> t = m.begin(o);
  CPPUNIT_ASSERT(* t ++ == 1);
  CPPUNIT_ASSERT(* t ++ == 2);
  CPPUNIT_ASSERT(* t ++ == 3);
  CPPUNIT_ASSERT(* t ++ == 4);
  CPPUNIT_ASSERT(* t ++ == 8);
  CPPUNIT_ASSERT(* t ++ == 9);
  CPPUNIT_ASSERT(t == m.end(o));

  /*
  m.put(1, v1, v1 + s1);
  m.put(2, v2, v2 + s2);
  m.put(3, v3, v3 + s3);
  map::range r1 = m.get(1);
  map::range r2 = m.get(2);
  map::range r3 = m.get(3);
  map::const_iterator i;

  CPPUNIT_ASSERT(r1.second - r1.first == s1);
  i = r1.first;
  CPPUNIT_ASSERT(*i++ == v1[ 0 ]);
  CPPUNIT_ASSERT(*i++ == v1[ 1 ]);
  CPPUNIT_ASSERT(*i++ == v1[ 2 ]);
  CPPUNIT_ASSERT(*i++ == v1[ 3 ]);
  CPPUNIT_ASSERT(i == r1.second);

  CPPUNIT_ASSERT(r2.second - r2.first == s2);
  i = r2.first;
  CPPUNIT_ASSERT(*i++ == v2[ 0 ]);
  CPPUNIT_ASSERT(*i++ == v2[ 1 ]);
  CPPUNIT_ASSERT(*i++ == v2[ 2 ]);
  CPPUNIT_ASSERT(i == r2.second);

  CPPUNIT_ASSERT(r3.second - r3.first == s3);
  i = r3.first;
  CPPUNIT_ASSERT(*i++ == v3[ 0 ]);
  CPPUNIT_ASSERT(*i++ == v3[ 1 ]);
  CPPUNIT_ASSERT(i == r3.second);

  CPPUNIT_ASSERT(m.size() == size_t(s1 + s2 + s3));


  */
  
}

#include <catch2/catch_all.hpp>
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/CopyPolicy.h"

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
    if ((d1 == 4 && d2 == 3) || (d1 == 3 && d2 == 4))
      return false;
    return d1 < d2;
  }
};

TEST_CASE("RangeMap", "[RangeMap]") {
  SECTION("checkAll") {
    typedef edm::RangeMap<int, std::vector<int>, edm::CopyPolicy<int> > map;

    map m;
    int v3[] = {1, 2, 3, 4};
    int v4[] = {10, 11, 12};
    int v2[] = {5, 6, 7};
    int v1[] = {8, 9};
    int s1 = sizeof(v1) / sizeof(int);
    int s2 = sizeof(v2) / sizeof(int);
    int s3 = sizeof(v3) / sizeof(int);
    int s4 = sizeof(v4) / sizeof(int);
    m.put(3, v3, v3 + s3);
    m.put(2, v2, v2 + s2);
    m.put(1, v1, v1 + s1);
    m.put(4, v4, v4 + s4);

    m.post_insert();

    map::const_iterator i = m.begin();
    REQUIRE(*i++ == 8);
    REQUIRE(*i++ == 9);
    REQUIRE(*i++ == 5);
    REQUIRE(*i++ == 6);
    REQUIRE(*i++ == 7);
    REQUIRE(*i++ == 1);
    REQUIRE(*i++ == 2);
    REQUIRE(*i++ == 3);
    REQUIRE(*i++ == 4);
    REQUIRE(*i++ == 10);
    REQUIRE(*i++ == 11);
    REQUIRE(*i++ == 12);
    REQUIRE(i == m.end());

    REQUIRE(*--i == 12);
    REQUIRE(*--i == 11);
    REQUIRE(*--i == 10);
    REQUIRE(*--i == 4);
    REQUIRE(*--i == 3);
    REQUIRE(*--i == 2);
    REQUIRE(*--i == 1);
    REQUIRE(*--i == 7);
    REQUIRE(*--i == 6);
    REQUIRE(*--i == 5);
    REQUIRE(*--i == 9);
    REQUIRE(*--i == 8);
    REQUIRE(i == m.begin());

    map::id_iterator b = m.id_begin();
    REQUIRE(m.id_size() == 4);
    REQUIRE(*b++ == 1);
    REQUIRE(*b++ == 2);
    REQUIRE(*b++ == 3);
    REQUIRE(*b++ == 4);
    REQUIRE(b == m.id_end());

    REQUIRE(*--b == 4);
    REQUIRE(*--b == 3);
    REQUIRE(*--b == 2);
    REQUIRE(*--b == 1);
    REQUIRE(b == m.id_begin());
    //
    // try the get
    //
    map::range r = m.get(1);
    REQUIRE(r.second - r.first == 2);
    r = m.get(2);
    REQUIRE(r.second - r.first == 3);
    r = m.get(3);
    REQUIRE(r.second - r.first == 4);
    r = m.get(4);
    REQUIRE(r.second - r.first == 3);

    //
    // try the get with comparator
    //
    r = m.get(3, IntComparator());
    REQUIRE(r.second - r.first == 7);
    i = r.first;

    REQUIRE(*i++ == 1);
    REQUIRE(*i++ == 2);
    REQUIRE(*i++ == 3);
    REQUIRE(*i++ == 4);
    REQUIRE(*i++ == 10);
    REQUIRE(*i++ == 11);
    REQUIRE(*i++ == 12);
    REQUIRE(i == r.second);

    r = m.get(1, IntComparator());
    REQUIRE(r.second - r.first == 2);
  }
}

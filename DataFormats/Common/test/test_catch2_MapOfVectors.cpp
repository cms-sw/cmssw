#include <vector>
#include <algorithm>
#include "catch2/catch_all.hpp"
#include "DataFormats/Common/interface/MapOfVectors.h"

typedef edm::MapOfVectors<int, int> MII;
typedef MII::TheMap TheMap;

class TestMapOfVectors {
public:
  static auto& keys(MII& m) { return m.m_keys; }
  static auto& offsets(const MII& m) { return m.m_offsets; }
  static auto& data(const MII& m) { return m.m_data; }
};

TEST_CASE("MapOfVectors", "[MapOfVectors]") {
  TheMap om;
  unsigned int tot = 0;
  int v[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (int i = 0; i < 10; ++i) {
    tot += i;
    om[i].resize(i);
    std::copy(v, v + i, om[i].begin());
  }

  SECTION("default ctor") {
    MII m;
    REQUIRE(m.size() == 0);
    REQUIRE(m.empty());
    REQUIRE(TestMapOfVectors::keys(m).size() == 0);
    REQUIRE(TestMapOfVectors::offsets(m).size() == 1);
    REQUIRE(TestMapOfVectors::offsets(m)[0] == 0);
    REQUIRE(TestMapOfVectors::data(m).size() == 0);
  }

  SECTION("filling") {
    MII m(om);
    REQUIRE(m.size() == om.size());
    REQUIRE(!m.empty());
    REQUIRE(TestMapOfVectors::keys(m).size() == om.size());
    REQUIRE(TestMapOfVectors::offsets(m).size() == om.size() + 1);
    REQUIRE(TestMapOfVectors::offsets(m)[0] == 0);
    REQUIRE(TestMapOfVectors::offsets(m)[m.size()] == tot);
    REQUIRE(TestMapOfVectors::data(m).size() == tot);
  }

  SECTION("find") {
    MII m(om);
    REQUIRE(m.find(-1) == m.emptyRange());
    for (TheMap::const_iterator p = om.begin(); p != om.end(); ++p) {
      MII::range r = m.find((*p).first);
      REQUIRE(int(r.size()) == (*p).first);
      REQUIRE(std::equal((*p).second.begin(), (*p).second.end(), r.begin()));
    }
  }

  SECTION("iterator") {
    MII m(om);
    TheMap::const_iterator op = om.begin();
    unsigned int lt = 0;
    for (MII::const_iterator p = m.begin(); p != m.end(); ++p) {
      REQUIRE((*p).first == (*op).first);
      REQUIRE(std::equal((*p).second.begin(), (*p).second.end(), (*op).second.begin()));
      lt += (*p).second.size();
      ++op;
    }
    REQUIRE(lt == tot);
  }
}

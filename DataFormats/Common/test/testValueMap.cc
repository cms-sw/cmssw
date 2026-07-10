#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

TEST_CASE("test ValueMap checkAll", "[ValueMap]") {
  typedef std::vector<int> CKey;
  typedef double Val;
  typedef edm::AssociationMap<edm::OneToValue<CKey, Val, unsigned char> > Assoc;
  Assoc v;
  REQUIRE(v.empty());
  REQUIRE(v.size() == 0);
}

// just check that some stuff compiles
namespace testValueMap {
  void dummy() {
    typedef std::vector<int> CKey;
    typedef double Val;
    typedef edm::AssociationMap<edm::OneToValue<CKey, Val, unsigned char> > Assoc;
    Assoc v;
    v.insert(edm::Ref<CKey>(), 3.145);
    v.insert(Assoc::value_type(edm::Ref<CKey>(), 3.145));
    Assoc::const_iterator b = v.begin(), e = v.end();
    ++b;
    ++e;
    Assoc::const_iterator f = v.find(edm::Ref<CKey>());
    v.numberOfAssociations(edm::Ref<CKey>());
    const double& x = v[edm::Ref<CKey>()];
    double y = x;
    ++y;
    std::cout << "numberOfAssociations:" << y << std::endl;
    ++f;
    edm::Ref<Assoc> r;
    v.erase(edm::Ref<CKey>());
    v.clear();
    REQUIRE(v.size() == 0);
    v.post_insert();
  }
}  // namespace testValueMap

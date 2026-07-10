#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

namespace testOneToOneAssociation {
  // just check that some stuff compiles
  void dummy() {
    typedef std::vector<int> CKey;
    typedef std::vector<double> CVal;
    typedef edm::AssociationMap<edm::OneToOne<CKey, CVal, unsigned char> > Assoc;
    Assoc v;
    REQUIRE(v.empty());
    REQUIRE(v.size() == 0);
    v.insert(edm::Ref<CKey>(), edm::Ref<CVal>());
    v.insert(Assoc::value_type(edm::Ref<CKey>(), edm::Ref<CVal>()));
    Assoc::const_iterator b = v.begin(), e = v.end();
    ++b;
    ++e;
    Assoc::const_iterator f = v.find(edm::Ref<CKey>());
    v.numberOfAssociations(edm::Ref<CKey>());
    const edm::Ref<CVal>& x = v[edm::Ref<CKey>()];
    x.id();
    ++f;
    int n = v.numberOfAssociations(edm::Ref<CKey>());
    ++n;
    std::cout << "numberOfAssociations:" << n << std::endl;
    edm::Ref<Assoc> r;
    v[edm::Ref<CKey>()];
    v.erase(edm::Ref<CKey>());
    v.clear();
    REQUIRE(v.size() == 0);
    v.post_insert();
  }
}  // namespace testOneToOneAssociation

TEST_CASE("OneToOneAssociation", "[OneToOneAssociation]") {
  SECTION("checkAll") {
    typedef std::vector<int> CKey;
    typedef std::vector<double> CVal;
    typedef edm::AssociationMap<edm::OneToOne<CKey, CVal, unsigned char> > Assoc;
    Assoc v;
    REQUIRE(v.empty());
    REQUIRE(v.size() == 0);

    // Call dummy to check compilation
    //dummy();
  }
}

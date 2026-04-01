#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

namespace testOneToManyAssociation {
  // just check that some stuff compiles
  void dummy() {
    typedef std::vector<int> CKey;
    typedef std::vector<double> CVal;
    {
      typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;
      Assoc v;
      v.insert(edm::Ref<CKey>(), edm::Ref<CVal>());
      v.insert(Assoc::value_type(edm::Ref<CKey>(), edm::RefVector<CVal>()));
      Assoc::const_iterator b = v.begin(), e = v.end();
      ++b;
      ++e;
      Assoc::const_iterator f = v.find(edm::Ref<CKey>());
      v.numberOfAssociations(edm::Ref<CKey>());
      const edm::RefVector<CVal>& x = v[edm::Ref<CKey>()];
      int n = x.size();
      ++f;
      n = v.numberOfAssociations(edm::Ref<CKey>());
      ++n;
      std::cout << "numberOfAssociations:" << n << std::endl;
      edm::Ref<Assoc> r;
      v[edm::Ref<CKey>()];
      v.erase(edm::Ref<CKey>());
      v.clear();
      REQUIRE(v.size() == 0);
      v.post_insert();
    }
    {
      typedef edm::AssociationMap<edm::OneToManyWithQuality<CKey, CVal, double, unsigned char> > Assoc;
      Assoc v;
      v.insert(edm::Ref<CKey>(), std::make_pair(edm::Ref<CVal>(), 3.14));
      Assoc::const_iterator b = v.begin(), e = v.end();
      ++b;
      ++e;
      Assoc::const_iterator f = v.find(edm::Ref<CKey>());
      v.numberOfAssociations(edm::Ref<CKey>());
      const std::vector<std::pair<edm::Ref<CVal>, double> >& x = v[edm::Ref<CKey>()];
      int n = x.size();
      ++f;
      n = v.numberOfAssociations(edm::Ref<CKey>());
      ++n;
      std::cout << "numberOfAssociations:" << n << std::endl;
      edm::Ref<Assoc> r;
      v[edm::Ref<CKey>()];
      v.erase(edm::Ref<CKey>());
      v.clear();
      REQUIRE(v.size() == 0);
      v.post_insert();
    }
  }
}  // namespace testOneToManyAssociation

TEST_CASE("OneToManyAssociation", "[OneToManyAssociation]") {
  SECTION("checkAll") {
    typedef std::vector<int> CKey;
    typedef std::vector<double> CVal;
    typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;
    Assoc v;
    REQUIRE(v.empty());
    REQUIRE(v.size() == 0);
  }
}

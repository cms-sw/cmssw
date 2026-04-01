#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/TestHandle.h"
using namespace edm;

TEST_CASE("test AssociationVector", "[AssociationVector]") {
  typedef std::vector<double> CKey;
  typedef std::vector<int> CVal;

  CKey k;
  k.push_back(1.1);
  k.push_back(2.2);
  k.push_back(3.3);
  ProductID const pid(1, 1);
  TestHandle<CKey> handle(&k, pid);
  RefProd<CKey> ref(handle);
  AssociationVector<RefProd<CKey>, CVal> v(ref);
  v.setValue(0, 1);
  v.setValue(1, 2);
  v.setValue(2, 3);
  REQUIRE(v.size() == 3);
  REQUIRE(v.keyProduct() == ref);
  REQUIRE(v[0].second == 1);
  REQUIRE(v[1].second == 2);
  REQUIRE(v[2].second == 3);
  REQUIRE(*v[0].first == 1.1);
  REQUIRE(*v[1].first == 2.2);
  REQUIRE(*v[2].first == 3.3);
  REQUIRE(*v.key(0) == 1.1);
  REQUIRE(*v.key(1) == 2.2);
  REQUIRE(*v.key(2) == 3.3);
  Ref<CKey> rc0(handle, 0), rc1(handle, 1), rc2(handle, 2);
  REQUIRE(v[rc0] == 1);
  REQUIRE(v[rc1] == 2);
  REQUIRE(v[rc2] == 3);
  ProductID const assocPid(1, 2);
  TestHandle<AssociationVector<RefProd<CKey>, CVal> > assocHandle(&v, assocPid);
  Ref<AssociationVector<RefProd<CKey>, CVal> > r1(assocHandle, 0);
  REQUIRE(*r1->first == 1.1);
  REQUIRE(r1->second == 1);
  Ref<AssociationVector<RefProd<CKey>, CVal> > r2(assocHandle, 1);
  REQUIRE(*r2->first == 2.2);
  REQUIRE(r2->second == 2);
  Ref<AssociationVector<RefProd<CKey>, CVal> > r3(assocHandle, 2);
  REQUIRE(*r3->first == 3.3);
  REQUIRE(r3->second == 3);
  v[rc0] = 111;
  v[rc1] = 122;
  v[rc2] = 133;
  REQUIRE(v[rc0] == 111);
  REQUIRE(v[rc1] == 122);
  REQUIRE(v[rc2] == 133);
}


#include <algorithm>

#include <catch2/catch_all.hpp>
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>

namespace testreftobase {
  struct Base {
    virtual ~Base() {}
    virtual int val() const = 0;
    bool operator==(Base const& other) const { return val() == other.val(); }
  };

  struct Inherit1 : public Base {
    virtual int val() const { return 1; }
  };
  struct Inherit2 : public Base {
    virtual int val() const { return 2; }
  };
}  // namespace testreftobase

using namespace testreftobase;

void do_some_tests(edm::RefToBaseVector<Base> x) {
  edm::RefToBaseVector<Base> copy(x);

  REQUIRE(x.empty() == copy.empty());
  REQUIRE(x.size() == copy.size());
  edm::RefToBaseVector<Base>::const_iterator b = x.begin(), e = x.end(), cb = copy.begin(), ce = copy.end();
  REQUIRE(e - b == ce - cb);
  REQUIRE(std::distance(b, e) == std::distance(cb, ce));
}

TEST_CASE("test RefToBaseVector", "[RefToBaseVector]") {
  using namespace edm;

  std::vector<Inherit1> v1(2, Inherit1());
  std::vector<Inherit2> v2(2, Inherit2());

  TestHandle<std::vector<Inherit1> > h1(&v1, ProductID(1, 1));
  RefVector<std::vector<Inherit1> > rv1;
  rv1.push_back(Ref<std::vector<Inherit1> >(h1, 0));
  rv1.push_back(Ref<std::vector<Inherit1> >(h1, 1));
  TestHandle<std::vector<Inherit2> > h2(&v2, ProductID(1, 2));
  RefVector<std::vector<Inherit2> > rv2;
  rv2.push_back(Ref<std::vector<Inherit2> >(h2, 0));
  rv2.push_back(Ref<std::vector<Inherit2> >(h2, 1));

  RefToBaseVector<Base> empty;
  RefToBaseVector<Base> copy_of_empty(empty);

  REQUIRE(empty == copy_of_empty);

  RefToBaseVector<Base> bv1(rv1);
  RefToBase<Base> r1_0 = bv1[0];
  RefToBase<Base> r1_1 = bv1[1];
  RefToBaseVector<Base> bv2(rv2);
  RefToBase<Base> r2_0 = bv2[0];
  RefToBase<Base> r2_1 = bv2[1];

  REQUIRE(bv1.empty() == false);
  REQUIRE(bv1.size() == 2);
  REQUIRE(bv2.size() == 2);
  REQUIRE(r1_0->val() == 1);
  REQUIRE(r1_1->val() == 1);
  REQUIRE(r2_0->val() == 2);
  REQUIRE(r2_1->val() == 2);

  RefToBaseVector<Base>::const_iterator b = bv1.begin(), e = bv1.end();
  RefToBaseVector<Base>::const_iterator i = b;
  REQUIRE((*i)->val() == 1);
  REQUIRE(i != e);
  REQUIRE(i - b == 0);
  ++i;
  REQUIRE((*i)->val() == 1);
  REQUIRE(i != e);
  REQUIRE(i - b == 1);
  ++i;
  REQUIRE(i == e);

  RefToBaseVector<Base> assigned_from_bv1;
  do_some_tests(assigned_from_bv1);
  REQUIRE(assigned_from_bv1.empty());
  assigned_from_bv1 = bv1;
  REQUIRE(assigned_from_bv1.size() == bv1.size());
  REQUIRE(std::equal(bv1.begin(), bv1.end(), assigned_from_bv1.begin()));
  REQUIRE(assigned_from_bv1 == bv1);

  do_some_tests(assigned_from_bv1);

  /// creation of empty vector adding with push_back
  RefToBaseVector<Base> bv3;
  bv3.push_back(r1_0);
  REQUIRE(bv3.size() == 1);
  REQUIRE(&(*r1_0) == &(*bv3[0]));
  bv3.push_back(r1_1);
  REQUIRE(bv3.size() == 2);
  REQUIRE(&(*r1_1) == &(*bv3[1]));
}


#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/IntValues.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {
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

  struct TestGetter : public edm::EDProductGetter {
    edm::WrapperBase const* hold_ = nullptr;
    edm::WrapperBase const* getIt(edm::ProductID const&) const override { return hold_; }
    unsigned int transitionIndex_() const override { return 0U; }

    TestGetter() : hold_() {}
  };
  void do_some_tests(edm::PtrVector<Base> const& x) {
    edm::PtrVector<Base> copy(x);

    REQUIRE(x.empty() == copy.empty());
    REQUIRE(x.size() == copy.size());
    edm::PtrVector<Base>::const_iterator b = x.begin(), e = x.end(), cb = copy.begin(), ce = copy.end();
    REQUIRE(e - b == ce - cb);
    REQUIRE(std::distance(b, e) == std::distance(cb, ce));
  }
}  // namespace

TEST_CASE("test PtrVector", "[PtrVector]") {
  using namespace edm;

  SECTION("check") {
    std::vector<Inherit1> v1(2, Inherit1());
    std::vector<Inherit2> v2(2, Inherit2());

    TestHandle<std::vector<Inherit1>> h1(&v1, ProductID(1, 1));
    PtrVector<Inherit1> rv1;
    rv1.push_back(Ptr<Inherit1>(h1, 0));
    rv1.push_back(Ptr<Inherit1>(h1, 1));
    TestHandle<std::vector<Inherit2>> h2(&v2, ProductID(1, 2));
    PtrVector<Inherit2> rv2;
    rv2.push_back(Ptr<Inherit2>(h2, 0));
    rv2.push_back(Ptr<Inherit2>(h2, 1));

    PtrVector<Base> empty;
    PtrVector<Base> copy_of_empty(empty);

    REQUIRE(empty == copy_of_empty);

    PtrVector<Base> bv1(rv1);
    Ptr<Base> r1_0 = bv1[0];
    Ptr<Base> r1_1 = bv1[1];
    PtrVector<Base> bv2(rv2);
    Ptr<Base> r2_0 = bv2[0];
    Ptr<Base> r2_1 = bv2[1];

    REQUIRE(bv1.empty() == false);
    REQUIRE(bv1.size() == 2);
    REQUIRE(bv2.size() == 2);
    REQUIRE(r1_0->val() == 1);
    REQUIRE(r1_1->val() == 1);
    REQUIRE(r2_0->val() == 2);
    REQUIRE(r2_1->val() == 2);

    PtrVector<Base>::const_iterator b = bv1.begin(), e = bv1.end();
    PtrVector<Base>::const_iterator i = b;
    REQUIRE((*i)->val() == 1);
    REQUIRE(i != e);
    REQUIRE(i - b == 0);
    ++i;
    REQUIRE((*i)->val() == 1);
    REQUIRE(i != e);
    REQUIRE(i - b == 1);
    ++i;
    REQUIRE(i == e);

    SECTION("assignment") {
      PtrVector<Base> assigned_from_bv1;
      do_some_tests(assigned_from_bv1);
      REQUIRE(assigned_from_bv1.empty());
      assigned_from_bv1 = bv1;
      REQUIRE(assigned_from_bv1.size() == bv1.size());
      REQUIRE(std::equal(bv1.begin(), bv1.end(), assigned_from_bv1.begin()));
      REQUIRE(assigned_from_bv1 == bv1);

      do_some_tests(assigned_from_bv1);
    }

    /// creation of empty vector adding with push_back
    SECTION("manipulate") {
      PtrVector<Base> bv3;
      bv3.push_back(r1_0);
      REQUIRE(bv3.size() == 1);
      REQUIRE(&(*r1_0) == &(*bv3[0]));
      bv3.push_back(r1_1);
      REQUIRE(bv3.size() == 2);
      REQUIRE(&(*r1_1) == &(*bv3[1]));

      /// clearing, then pushing in Ptr with other product ID
      bv3.clear();
      REQUIRE(bv3.size() == 0);
      bv3.push_back(r2_0);
      REQUIRE(bv3.size() == 1);
    }
  }
  SECTION("get") {
    using namespace test_with_dictionaries;
    typedef std::vector<IntValue> IntCollection;
    auto ptr = std::make_unique<IntCollection>();

    ptr->push_back(0);
    ptr->push_back(1);
    ptr->push_back(2);
    ptr->push_back(3);

    edm::Wrapper<IntCollection> wrapper(std::move(ptr));
    TestGetter tester;
    SECTION("not available") {
      edm::ProductID const pid(1, 1);
      edm::PtrVector<IntValue> iVec;
      iVec.push_back(edm::Ptr<IntValue>(pid, 0, &tester));
      iVec.push_back(edm::Ptr<IntValue>(pid, 2, &tester));

      iVec.setProductGetter(&tester);

      REQUIRE(iVec.size() == 2);
      REQUIRE(iVec.isAvailable() == false);
    }
    tester.hold_ = &wrapper;

    edm::ProductID const pid(1, 1);

    IntCollection const* wptr = dynamic_cast<IntCollection const*>(wrapper.product());

    edm::PtrVector<IntValue> iVec;
    iVec.push_back(edm::Ptr<IntValue>(pid, 0, &tester));
    iVec.push_back(edm::Ptr<IntValue>(pid, 2, &tester));

    iVec.setProductGetter(&tester);

    REQUIRE(iVec.size() == 2);
    REQUIRE(static_cast<size_t>(iVec.end() - iVec.begin()) == iVec.size());
    REQUIRE(&(*(*(iVec.begin()))) == &(*(wptr->begin())));
    REQUIRE((*(*(iVec.begin()))).value_ == 0);
    REQUIRE((*(iVec.begin()))->value_ == 0);
    SECTION("available") { REQUIRE(iVec.isAvailable() == true); }
  }
}

#include <algorithm>
#include <cassert>
#include <memory>
#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace ownvector_test {
  struct Base {
    virtual ~Base();
    virtual Base* clone() const = 0;
  };

  Base::~Base() {}

  struct Derived : Base {
    explicit Derived(int n);
    Derived(Derived const& other);
    Derived& operator=(Derived const& other);
    virtual ~Derived();
    void swap(Derived& other);
    virtual Derived* clone() const;

    edm::propagate_const<int*> pointer;
  };

  void Derived::swap(Derived& other) { std::swap(pointer, other.pointer); }
  Derived::Derived(int n) : pointer(new int(n)) {}

  Derived::Derived(Derived const& other) : pointer(new int(*other.pointer)) {}

  Derived& Derived::operator=(Derived const& other) {
    Derived temp(other);
    swap(temp);
    return *this;
  }

  Derived::~Derived() { delete pointer.get(); }

  Derived* Derived::clone() const { return new Derived(*this); }

  void swap(Derived& a, Derived& b) { a.swap(b); }

  void do_assign(edm::OwnVector<Base>& iLHS, edm::OwnVector<Base>& iRHS) { iLHS = iRHS; }

}  // namespace ownvector_test

using namespace ownvector_test;

TEST_CASE("test OwnVector", "[OwnVector]") {
  SECTION("copy_good_vec") {
    // v1 is perfectly fine...
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    v1.push_back(p);
    //v1.push_back(new Derived(100));

    // But what if we copy him?
    edm::OwnVector<Base> v2(v1);
  }

  SECTION("assign_to_other") {
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    v1.push_back(p);

    edm::OwnVector<Base> v2;
    v2 = v1;
  }

  SECTION("assign_to_self") {
    // Self-assignment happens, often by accident...
    edm::OwnVector<Base> v1;
    v1.push_back(new Derived(100));
    auto& v2 = v1;
    do_assign(v1, v2);
  }

  SECTION("pop_one") {
    edm::OwnVector<Base> v1;
    v1.push_back(new Derived(100));
    v1.pop_back();
  }

  SECTION("back_with_null_pointer") {
    edm::OwnVector<Base> v;
    Base* p = 0;
    v.push_back(p);
    REQUIRE_THROWS_AS(v.back(), edm::Exception);
  }

  SECTION("take_an_rvalue") {
    edm::OwnVector<Base> v;
    v.push_back(new Derived(101));
    Derived d(102);
    v.push_back(d.clone());
  }

  SECTION("take_an_lvalue") {
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    v1.push_back(p);

    REQUIRE(p == nullptr);
  }

  SECTION("take_an_auto_ptr") {
    edm::OwnVector<Base> v1;
    std::unique_ptr<Base> p = std::make_unique<Derived>(100);
    v1.push_back(std::move(p));
    REQUIRE(p.get() == nullptr);
  }

  SECTION("set_at_index") {
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    Base* backup = p;
    v1.push_back(p);
    REQUIRE(p == nullptr);
    REQUIRE(&v1[0] == backup);
    Base* p2 = new Derived(101);
    Base* backup2 = p2;
    REQUIRE(backup2 != backup);
    v1.set(0, p2);
    REQUIRE(p2 == nullptr);
    REQUIRE(&v1[0] == backup2);
  }

  SECTION("insert_with_iter") {
    edm::OwnVector<Base> v1;
    Base *p[3], *backup[3];
    for (int i = 0; i < 3; ++i) {
      backup[i] = p[i] = new Derived(100 + i);
    }
    v1.push_back(p[0]);
    v1.push_back(p[2]);
    v1.insert(v1.begin() + 1, p[1]);
    REQUIRE(p[0] == nullptr);
    REQUIRE(p[1] == nullptr);
    REQUIRE(p[2] == nullptr);
    REQUIRE(&v1[0] == backup[0]);
    REQUIRE(&v1[1] == backup[1]);
    REQUIRE(&v1[2] == backup[2]);
  }
  SECTION("no crash") {
    edm::OwnVector<Base> vec;
    vec.push_back(new Derived(100));
    edm::OwnVector<Base>* p = new edm::OwnVector<Base>;
    p->push_back(new Derived(2));
    delete p;
  }
}

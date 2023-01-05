#include "catch.hpp"
#include "FWCore/Utilities/interface/clone_ptr.h"
#include <vector>

namespace {

  struct A {
    A() {}
    A(int j) : i(j) {}

    A *clone() const {
      cla++; /*std::cout<< "c A " << i << std::endl;*/
      return new A(*this);
    }

    int i = 3;
    static int cla;
    struct Guard {
      Guard() : old_(A::cla) { A::cla = 0; }
      ~Guard() { A::cla = old_; }
      int old_;
    };
  };
  int A::cla = 0;

  struct B {
    B() {}
    B(B const &b) : a(b.a) {}
    B(B &&b) noexcept : a(std::move(b.a)) {}

    B &operator=(B const &b) {
      a = b.a;
      return *this;
    }
    B &operator=(B &&b) noexcept {
      a = std::move(b.a);
      return *this;
    }

    extstd::clone_ptr<A> a;
  };

}  // namespace

TEST_CASE("Test clone_ptr", "[clone_ptr]") {
  SECTION("standalone") {
    A::Guard ag;
    B b;
    b.a.reset(new A(2));

    REQUIRE(A::cla == 0);
    B c = b;
    REQUIRE(A::cla == 1);
    B d = b;
    REQUIRE(A::cla == 2);

    SECTION("separate values") {
      b.a.reset(new A(-2));
      REQUIRE(c.a->i == 2);
      REQUIRE(b.a->i == -2);
      SECTION("operator=") {
        A::Guard ag;
        c = b;
        REQUIRE(A::cla == 1);

        REQUIRE(c.a->i == -2);
        c.a.reset(new A(-7));
        REQUIRE(c.a->i == -7);
        REQUIRE(A::cla == 1);
      }
    }
  }
  SECTION("in vector") {
    A::Guard ag;
    std::vector<B> vb(1);
    B b;
    b.a.reset(new A(-2));
    B c;
    c.a.reset(new A(-7));
    B d;
    d.a.reset(new A(2));
    SECTION("push_back") {
      A::Guard ag;
      vb.push_back(b);
      REQUIRE(A::cla == 1);

      int cValue = c.a->i;
      vb.push_back(std::move(c));
      //some compilers will still cause a clone to be called and others won't
      REQUIRE((*vb.back().a).i == cValue);
      SECTION("assign to element") {
        A::Guard ag;
        vb[0] = d;

        REQUIRE(A::cla == 1);
        REQUIRE(vb[0].a->i == d.a->i);
        REQUIRE(vb[0].a->i == 2);
        SECTION("sort") {
          A::Guard ag;
          std::sort(vb.begin(), vb.end(), [](B const &rh, B const &lh) { return rh.a->i < lh.a->i; });
          REQUIRE((*vb[0].a).i == -7);
          REQUIRE(A::cla == 0);
          std::swap(b, d);
          REQUIRE(A::cla == 0);
        }
      }
    }
  }
}

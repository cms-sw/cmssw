#include "catch.hpp"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <memory>
#include <atomic>

namespace {
  class simple {
    int i;

  public:
    static std::atomic<int> count;
    static std::atomic<int> count1;
    simple() : i(0) {
      ++count;
      ++count1;
    }
    explicit simple(int j) : i(j) {
      ++count;
      ++count1;
    }
    simple(simple const& s) : i(s.i) {
      ++count;
      ++count1;
    }
    ~simple() { --count; }
    bool operator==(simple const& o) const { return i == o.i; }
    bool isSame(simple const& o) const { return &o == this; }
    simple& operator=(simple const& o) {
      simple temp(o);
      std::swap(*this, temp);
      return *this;
    }
  };

  std::atomic<int> simple::count{0};
  std::atomic<int> simple::count1{0};
}  // namespace
TEST_CASE("test edm::value_ptr", "[value_ptr]") {
  REQUIRE(simple::count == 0);
  {
    edm::value_ptr<simple> a(new simple(10));
    REQUIRE(simple::count == 1);
    edm::value_ptr<simple> b(a);
    REQUIRE(simple::count == 2);

    REQUIRE(*a == *b);
    REQUIRE(a->isSame(*b) == false);
  }  // a and b destroyed
  REQUIRE(simple::count == 0);

  {
    auto c = std::make_unique<simple>(11);
    auto d = std::make_unique<simple>(11);
    REQUIRE(c.get() != nullptr);
    REQUIRE(d.get() != nullptr);
    simple* pc = c.get();
    simple* pd = d.get();

    edm::value_ptr<simple> e(std::move(c));
    REQUIRE(c.get() == nullptr);
    REQUIRE(*d == *e);
    REQUIRE(e.operator->() == pc);

    edm::value_ptr<simple> f;
    REQUIRE(not bool(f));
    f = std::move(d);
    REQUIRE(d.get() == nullptr);
    REQUIRE(*e == *f);
    REQUIRE(f.operator->() == pd);
    REQUIRE(bool(f));

    REQUIRE(simple::count == 2);
    REQUIRE(simple::count1 == 4);
  }
  REQUIRE(simple::count == 0);
  REQUIRE(simple::count1 == 4);
}

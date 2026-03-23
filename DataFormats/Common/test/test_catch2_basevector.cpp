#include "DataFormats/Common/interface/BaseVector.h"
#include "catch2/catch_all.hpp"

namespace {
  struct Base {
    virtual ~Base() = default;
    virtual int value() const = 0;
  };
  struct DerivedA : public Base {
    int value() const override { return 42; }
  };
  struct DerivedB : public Base {
    int value() const override { return 84; }
  };
}  // namespace
namespace edm {
  template <>
  struct BaseVectorTraits<Base> {
    using variant_type = std::variant<DerivedA, DerivedB>;
  };
}  // namespace edm
TEST_CASE("BaseVector can hold different types of objects", "[BaseVector]") {
  edm::BaseVector<Base> bv;
  SECTION("empty") {
    REQUIRE(bv.empty());
    REQUIRE(bv.size() == 0);
  }
  SECTION("emplace_back") {
    bv.emplace_back<DerivedA>();
    bv.emplace_back<DerivedB>();
    REQUIRE(bv.size() == 2);

    SECTION("operator[]") {
      REQUIRE(bv[0].value() == 42);
      REQUIRE(bv[1].value() == 84);
    }
  }

  SECTION("front/back") {
    bv.emplace_back<DerivedA>();
    bv.emplace_back<DerivedB>();
    REQUIRE(bv.front().value() == 42);
    REQUIRE(bv.back().value() == 84);
  }
  SECTION("emplace") {
    auto it = bv.emplace<DerivedA>(bv.end());
    REQUIRE(it->value() == 42);
    REQUIRE(bv.back().value() == 42);
    it = bv.emplace<DerivedB>(bv.begin());
    REQUIRE(it->value() == 84);
    REQUIRE(bv.front().value() == 84);
  }
  SECTION("insert") {
    auto it = bv.insert(bv.end(), DerivedA{});
    REQUIRE(it->value() == 42);
    REQUIRE(bv.back().value() == 42);
    it = bv.insert(bv.begin(), DerivedB{});
    REQUIRE(it->value() == 84);
    REQUIRE(bv.front().value() == 84);
  }

  SECTION("push_back") {
    bv.push_back(DerivedA{});
    bv.push_back(DerivedB{});
    REQUIRE(bv.size() == 2);

    SECTION("operator[]") {
      REQUIRE(bv[0].value() == 42);
      REQUIRE(bv[1].value() == 84);
    }
  }

  SECTION("get_if") {
    bv.push_back(DerivedA{});
    bv.push_back(DerivedB{});
    auto a = bv.get_if<DerivedA>(0);
    auto b = bv.get_if<DerivedB>(1);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(a->value() == 42);
    REQUIRE(b->value() == 84);
  }
  SECTION("iterator") {
    bv.push_back(DerivedA{});
    bv.push_back(DerivedB{});
    auto it = bv.begin();
    REQUIRE(bv.begin() != bv.end());
    REQUIRE(it->value() == 42);
    REQUIRE(it + 2 == bv.end());
    REQUIRE(bv.end() - 2 == bv.begin());
    REQUIRE(bv.end() - bv.begin() == 2);
    ++it;
    REQUIRE(it->value() == 84);
    ++it;
    REQUIRE(it == bv.end());
  }
  SECTION("const iterator") {
    bv.push_back(DerivedA{});
    bv.push_back(DerivedB{});
    auto it = bv.cbegin();
    REQUIRE(bv.cbegin() != bv.cend());
    REQUIRE(it + 2 == bv.end());
    REQUIRE(bv.end() - 2 == bv.begin());
    REQUIRE(bv.end() - bv.begin() == 2);
    REQUIRE(it->value() == 42);
    ++it;
    REQUIRE(it->value() == 84);
    ++it;
    REQUIRE(it == bv.cend());
  }
  SECTION("variant iterators") {
    bv.push_back(DerivedA{});
    bv.push_back(DerivedB{});
    auto it = bv.variant_begin();
    REQUIRE(it != bv.variant_end());
    REQUIRE(std::holds_alternative<DerivedA>(*it));
    REQUIRE(std::get<DerivedA>(*it).value() == 42);
    ++it;
    REQUIRE(std::holds_alternative<DerivedB>(*it));
    REQUIRE(std::get<DerivedB>(*it).value() == 84);
    ++it;
    REQUIRE(it == bv.variant_end());
    SECTION("sort with variant iterators") {
      //The following fails as one can't allocate a Base
      //    std::sort(bv.begin(), bv.end(), [](Base const& a, Base const& b) { return a.value() > b.value(); });
      //But sorting the underlying variant works as the variant holds the actual type and not the base class
      std::sort(bv.variant_begin(), bv.variant_end(), [](auto const& a, auto const& b) {
        return std::visit([](auto const& v) { return v.value(); }, a) >
               std::visit([](auto const& v) { return v.value(); }, b);
      });
      REQUIRE(bv[0].value() == 84);
      REQUIRE(bv[1].value() == 42);
    }
  }
}

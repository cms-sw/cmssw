#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/View.h"

typedef int value_type;
typedef edm::View<value_type> View;
typedef View::const_iterator const_iterator;
typedef View::const_reverse_iterator const_reverse_iterator;
typedef View::const_reference const_reference;
typedef View::size_type size_type;

TEST_CASE("View", "[Framework]") {
  SECTION("basic") {
    View v1;
    REQUIRE(v1.size() == 0);
    REQUIRE(v1.empty());
    View v2(v1);
    REQUIRE(v1 == v2);
    View v3;
    v3 = v1;
    REQUIRE(v3 == v1);

    REQUIRE(!(v1 < v2));
    REQUIRE(v1 <= v2);
    REQUIRE(!(v1 > v2));
    REQUIRE(v1 >= v2);
    REQUIRE(!(v1 != v2));
  }

  SECTION("createFromArray") {
    value_type vals[] = {1, 2, 3, 4, 5};
    size_t sz = sizeof(vals) / sizeof(value_type);

    View v1;
    edm::View<int>::fill_from_range(vals, vals + sz, v1);
    REQUIRE(v1.size() == 5);
    View v2;
    edm::View<int>::fill_from_range(vals, vals + sz, v2);
    REQUIRE(v1 == v2);
  }

  SECTION("directAccess") {
    value_type vals[] = {1, 2, 3, 4, 5};
    size_t sz = sizeof(vals) / sizeof(value_type);

    View v1;
    edm::View<int>::fill_from_range(vals, vals + sz, v1);
    for (size_type i = 0; i < v1.size(); ++i) {
      REQUIRE(v1[i] == vals[i]);
    }
  }

  SECTION("iterateForward") {
    value_type vals[] = {1, 2, 3, 4, 5};
    size_t sz = sizeof(vals) / sizeof(value_type);

    View v1;
    edm::View<int>::fill_from_range(vals, vals + sz, v1);

    const_iterator i = v1.begin();
    REQUIRE(*i == 1);
    ++i;
    REQUIRE(*i == 2);
  }

  SECTION("iterateBackward") {
    value_type vals[] = {1, 2, 3, 4, 5};
    size_t sz = sizeof(vals) / sizeof(value_type);

    View v1;
    edm::View<int>::fill_from_range(vals, vals + sz, v1);

    const_reverse_iterator i = v1.rbegin();
    REQUIRE(*i == 5);
    ++i;
    REQUIRE(*i == 4);
  }

  SECTION("cloning") {
    value_type vals[] = {1, 2, 3, 4, 5};
    size_t sz = sizeof(vals) / sizeof(value_type);

    View v1;
    edm::View<int>::fill_from_range(vals, vals + sz, v1);

    auto base = v1.clone();
    REQUIRE(base);
    edm::View<int>* view = dynamic_cast<edm::View<int>*>(base.get());
    REQUIRE(view);
    if (view) {
      REQUIRE(*view == v1);
    }
  }

  SECTION("ptrs") {
    value_type vals[] = {1, 2, 3, 4, 5};
    size_t sz = sizeof(vals) / sizeof(value_type);

    View v1;
    edm::View<int>::fill_from_range(vals, vals + sz, v1);
    size_t i = 0;
    for (auto ptr : v1.ptrs()) {
      REQUIRE(*ptr == vals[i++]);
    }
  }

}  // TEST_CASE

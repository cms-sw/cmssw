#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "SimpleEDProductGetter.h"

TEST_CASE("RefCore", "[RefCore]") {
  SECTION("default constructor") {
    edm::RefCore default_refcore;
    REQUIRE(default_refcore.isNull());
    REQUIRE(default_refcore.isNonnull() == false);
    REQUIRE(!default_refcore);
    REQUIRE(default_refcore.productGetter() == 0);
    REQUIRE(default_refcore.id().isValid() == false);
  }

  SECTION("non-default constructor") {
    SimpleEDProductGetter getter;
    edm::ProductID id(1, 201U);
    REQUIRE(id.isValid());
    edm::RefCore refcore(id, 0, &getter, false);
    REQUIRE(refcore.isNull() == false);
    REQUIRE(refcore.isNonnull());
    REQUIRE(!!refcore);
    REQUIRE(refcore.productGetter() == &getter);
    REQUIRE(refcore.id().isValid());
  }
}

TEST_CASE("RefCoreWithIndex", "[RefCore]") {
  SECTION("default constructor") {
    edm::RefCoreWithIndex default_refcore;
    REQUIRE(default_refcore.isNull());
    REQUIRE(default_refcore.isNonnull() == false);
    REQUIRE(!default_refcore);
    REQUIRE(default_refcore.productGetter() == 0);
    REQUIRE(default_refcore.id().isValid() == false);
    REQUIRE(default_refcore.index() == edm::key_traits<unsigned int>::value);
    edm::RefCore compareTo;
    edm::RefCore const& converted = default_refcore.toRefCore();
    REQUIRE(compareTo.productGetter() == converted.productGetter());
    REQUIRE(compareTo.id() == converted.id());
  }

  SECTION("non-default constructor") {
    SimpleEDProductGetter getter;
    edm::ProductID id(1, 201U);
    REQUIRE(id.isValid());
    edm::RefCoreWithIndex refcore(id, 0, &getter, false, 1);
    REQUIRE(refcore.isNull() == false);
    REQUIRE(refcore.isNonnull());
    REQUIRE(!!refcore);
    REQUIRE(refcore.productGetter() == &getter);
    REQUIRE(refcore.id().isValid());
    REQUIRE(refcore.index() == 1);
    edm::RefCore compareTo(id, 0, &getter, false);
    edm::RefCore const& converted = refcore.toRefCore();
    REQUIRE(compareTo.productGetter() == converted.productGetter());
    REQUIRE(compareTo.id() == converted.id());
  }
}

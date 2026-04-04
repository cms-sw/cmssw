/*
 *  datakey_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/31/05.
 *  Changed by Viji Sundararajan on 24-Jun-2005.
 */

#include "catch2/catch_all.hpp"
#include <cstring>
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/HCTypeTag.h"

using namespace edm;
using namespace edm::eventsetup;

namespace datakey_t {
  class Dummy {};
  class Dummy2 {};
}  // namespace datakey_t
using datakey_t::Dummy;
using datakey_t::Dummy2;

HCTYPETAG_HELPER_METHODS(Dummy)
HCTYPETAG_HELPER_METHODS(Dummy2)

TEST_CASE("DataKey", "[Framework][EventSetup]") {
  SECTION("nametagConstructionTest") {
    const NameTag defaultTag;
    REQUIRE(0 == std::strcmp("", defaultTag.value()));

    const NameTag namedTag("fred");
    REQUIRE(0 == std::strcmp("fred", namedTag.value()));
  }

  SECTION("nametagComparisonTest") {
    const NameTag defaultTag;
    REQUIRE(defaultTag == defaultTag);

    const NameTag fredTag("fred");
    REQUIRE(fredTag == fredTag);

    REQUIRE(!(defaultTag == fredTag));

    const NameTag barneyTag("barney");

    REQUIRE(barneyTag < fredTag);
  }

  SECTION("nametagCopyTest") {
    const NameTag defaultTag;
    NameTag tester(defaultTag);
    REQUIRE(tester == defaultTag);

    const NameTag fredTag("fred");
    tester = fredTag;
    REQUIRE(tester == fredTag);
  }

  SECTION("ConstructionTest") {
    DataKey defaultKey;
    REQUIRE(TypeTag() == defaultKey.type());
    REQUIRE(0 == std::strcmp("", defaultKey.name().value()));

    DataKey dummyKey(DataKey::makeTypeTag<Dummy>(), "");
    REQUIRE(DataKey::makeTypeTag<Dummy>() == dummyKey.type());
    REQUIRE(0 == std::strcmp("", dummyKey.name().value()));

    DataKey namedDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
    REQUIRE(DataKey::makeTypeTag<Dummy>() == namedDummyKey.type());
    REQUIRE(0 == std::strcmp("fred", namedDummyKey.name().value()));
  }

  SECTION("ComparisonTest") {
    const DataKey defaultKey;
    REQUIRE(defaultKey == defaultKey);
    REQUIRE(!(defaultKey < defaultKey));

    const DataKey dummyKey(DataKey::makeTypeTag<Dummy>(), "");
    const DataKey fredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
    const DataKey barneyDummyKey(DataKey::makeTypeTag<Dummy>(), "barney");

    REQUIRE(!(defaultKey == dummyKey));
    REQUIRE(dummyKey == dummyKey);
    REQUIRE(!(dummyKey == fredDummyKey));

    REQUIRE(barneyDummyKey == barneyDummyKey);
    REQUIRE(barneyDummyKey < fredDummyKey);
    REQUIRE(!(fredDummyKey < barneyDummyKey));
    REQUIRE(!(barneyDummyKey == fredDummyKey));

    const DataKey dummy2Key(DataKey::makeTypeTag<Dummy2>(), "");

    REQUIRE(!(dummy2Key == dummyKey));
  }

  SECTION("CopyTest") {
    const DataKey defaultKey;
    DataKey tester(defaultKey);
    REQUIRE(tester == defaultKey);

    const DataKey dummyKey(DataKey::makeTypeTag<Dummy>(), "");
    tester = dummyKey;
    REQUIRE(tester == dummyKey);
    const DataKey fredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
    tester = fredDummyKey;
    REQUIRE(tester == fredDummyKey);

    DataKey tester2(fredDummyKey);
    REQUIRE(tester2 == fredDummyKey);
  }

  SECTION("nocopyConstructionTest") {
    const DataKey fredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
    const DataKey noCopyFredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred", DataKey::kDoNotCopyMemory);

    REQUIRE(fredDummyKey == noCopyFredDummyKey);

    const DataKey copyFredDummyKey(noCopyFredDummyKey);
    REQUIRE(copyFredDummyKey == noCopyFredDummyKey);

    DataKey copy2FredDummyKey;
    copy2FredDummyKey = noCopyFredDummyKey;
    REQUIRE(copy2FredDummyKey == noCopyFredDummyKey);

    DataKey noCopyBarneyDummyKey(DataKey::makeTypeTag<Dummy>(), "barney", DataKey::kDoNotCopyMemory);

    noCopyBarneyDummyKey = noCopyFredDummyKey;
    REQUIRE(noCopyBarneyDummyKey == noCopyFredDummyKey);
  }
}

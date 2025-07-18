#include "catch.hpp"

#include "DataFormats/Provenance/interface/CompactHash.h"
#include "FWCore/Utilities/interface/Digest.h"

namespace {
  using TestHash = edm::CompactHash<100>;
}

TEST_CASE("CompactHash", "[CompactHash]") {
  SECTION("Default construction is invalid") { REQUIRE(TestHash{}.isValid() == false); }

  SECTION("Basic operations") {
    cms::Digest d("foo");
    auto result = d.digest().bytes;

    TestHash id{result};
    REQUIRE(id.isValid() == true);
    REQUIRE(id.compactForm() == result);

    TestHash id2 = id;
    REQUIRE(id2.isValid() == true);
    REQUIRE(id2.compactForm() == result);

    cms::Digest b("bar");
    auto diffResult = b.digest().bytes;
    REQUIRE(id2 == TestHash{result});
    REQUIRE(id2 != TestHash{diffResult});

    REQUIRE(id2 > TestHash{diffResult});
    REQUIRE(TestHash{diffResult} < id2);

    REQUIRE(not(id2 < id2));
  }
}

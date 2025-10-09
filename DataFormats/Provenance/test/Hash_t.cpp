#include "catch2/catch_all.hpp"

#include <algorithm>
#include <ranges>
#include <vector>

#include "DataFormats/Provenance/interface/Hash.h"
#include "FWCore/Utilities/interface/Digest.h"

namespace {
  using TestHash = edm::Hash<100>;
}

TEST_CASE("Hash", "[Hash]") {
  SECTION("Default construction is invalid") { REQUIRE(TestHash{}.isValid() == false); }

  SECTION("Basic operations") {
    cms::Digest d("foo");
    auto result = d.digest().toString();

    TestHash id{result};
    REQUIRE(id.isValid() == true);
    {
      std::string idString;
      id.toString(idString);
      REQUIRE(idString == result);
    }

    TestHash id2 = id;
    REQUIRE(id2.isValid() == true);
    {
      std::string id2String;
      id2.toString(id2String);
      REQUIRE(id2String == result);
    }

    cms::Digest b("bar");
    auto diffResult = b.digest().toString();
    REQUIRE(id2 == TestHash{result});
    REQUIRE(id2 != TestHash{diffResult});

    REQUIRE(id2 > TestHash{diffResult});
    REQUIRE(TestHash{diffResult} < id2);

    REQUIRE(not(id2 < id2));
  }

  SECTION("std::ranges::sort") {
    std::vector<TestHash> container{TestHash{cms::Digest("foo").digest().toString()},
                                    TestHash{cms::Digest("bar").digest().toString()},
                                    TestHash{cms::Digest("fred").digest().toString()},
                                    TestHash{cms::Digest("wilma").digest().toString()}};
    CHECK(container[0] > container[1]);
    CHECK(container[1] < container[2]);
    CHECK(container[2] > container[3]);

    std::ranges::sort(container);
    CHECK(container[0] < container[1]);
    CHECK(container[1] < container[2]);
    CHECK(container[2] < container[3]);
  }
}

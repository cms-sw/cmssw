#include <functional>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "catch.hpp"

namespace edm {
  class TestEDGetToken {
  public:
    template <typename... Args>
    static edm::EDGetToken makeToken(Args&&... iArgs) {
      return edm::EDGetToken(std::forward<Args>(iArgs)...);
    }

    template <typename T, typename... Args>
    static edm::EDGetTokenT<T> makeTokenT(Args&&... iArgs) {
      return edm::EDGetTokenT<T>(std::forward<Args>(iArgs)...);
    }

    template <typename T, typename... Args>
    static const edm::EDGetTokenT<T> makeConstTokenT(Args&&... iArgs) {
      return edm::EDGetTokenT<T>(std::forward<Args>(iArgs)...);
    }
  };
}  // namespace edm

namespace {
  class Adapter {
  public:
    template <typename T>
    edm::EDGetTokenT<T> consumes() const {
      return edm::TestEDGetToken::makeTokenT<T>(11U);
    }
  };
}  // namespace

TEST_CASE("Test EDGetToken", "[EDGetToken]") {
  SECTION("EDGetTokenT") {
    SECTION("No argument ctr") {
      edm::EDGetTokenT<int> token1 = edm::TestEDGetToken::makeTokenT<int>();
      REQUIRE(token1.isUninitialized());
      REQUIRE((token1.index() == 0xFFFFFFFF));
    }
    SECTION("1 arg ctr") {
      edm::EDGetTokenT<int> token2 = edm::TestEDGetToken::makeTokenT<int>(11U);
      REQUIRE(!token2.isUninitialized());
      REQUIRE((token2.index() == 11));
    }
    SECTION("non const arg copy ctr") {
      edm::EDGetTokenT<int> token2 = edm::TestEDGetToken::makeTokenT<int>(11U);
      edm::EDGetTokenT<int> token3(token2);
      REQUIRE(!token3.isUninitialized());
      REQUIRE((token3.index() == 11));
    }

    SECTION("const arg copy ctr") {
      const edm::EDGetTokenT<int> cToken2 = edm::TestEDGetToken::makeTokenT<int>(11U);
      edm::EDGetTokenT<int> token4(cToken2);
      REQUIRE(!token4.isUninitialized());
      REQUIRE(token4.index() == 11);
    }

    SECTION("const arg copy ctr with move") {
      auto const t = edm::TestEDGetToken::makeConstTokenT<int>(11U);
      const edm::EDGetTokenT<int> cToken2{std::move(t)};
      REQUIRE(!cToken2.isUninitialized());
      REQUIRE(cToken2.index() == 11);
    }

    SECTION("Use Adapter") {
      Adapter a;
      const edm::EDGetTokenT<int> cToken2{a};
      REQUIRE(!cToken2.isUninitialized());
      REQUIRE(cToken2.index() == 11);
    }
  }

  SECTION("EDGetToken") {
    SECTION("No arg ctr") {
      edm::EDGetToken token10 = edm::TestEDGetToken::makeToken();
      REQUIRE(token10.isUninitialized());
      REQUIRE(token10.index() == 0xFFFFFFFF);
    }

    SECTION("1 arg ctr") {
      edm::EDGetToken token11 = edm::TestEDGetToken::makeToken(100);
      REQUIRE(!token11.isUninitialized());
      REQUIRE(token11.index() == 100);
    }

    SECTION("EDGetTokenT to EDGetToken ctr") {
      edm::EDGetTokenT<int> token2 = edm::TestEDGetToken::makeTokenT<int>(11U);
      edm::EDGetToken token12(token2);
      REQUIRE(!token12.isUninitialized());
      REQUIRE(token12.index() == 11);
    }
  }
}

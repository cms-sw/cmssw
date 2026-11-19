#include "catch2/catch_all.hpp"

#include "FWCore/Utilities/src/pathBeginsWith.h"

#include <filesystem>

TEST_CASE("pathBeginsWith", "[FileInPath]") {
  namespace fs = std::filesystem;

  SECTION("pathBeginsWith returns true for a path that begins with the prefix") {
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/a/b")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/a/b/")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/a")));
  }

  SECTION("pathBeginsWith returns false for a path that does not begin with the prefix") {
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/a/b/c/d")));
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/x/y/z")));
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("")));
  }

  SECTION("pathBeginsWith handles empty paths correctly") {
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path(""), fs::path("/a/b/c")));
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path(""), fs::path("")));
  }

  SECTION("pathBeginsWith handles relative paths correctly") {
    REQUIRE(edm::detail::pathBeginsWith(fs::path("a/b/c"), fs::path("a/b")));
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("a/b/c"), fs::path("x/y/z")));
  }

  SECTION("pathBeginsWith handles paths with .. and . correctly") {
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/./c"), fs::path("/a/b")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/a/./b")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/./c"), fs::path("/a/./b")));

    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/e/../b/c"), fs::path("/a/b")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/b/c"), fs::path("/a/d/../b")));
    REQUIRE(edm::detail::pathBeginsWith(fs::path("/a/e/../b/c"), fs::path("/a/d/../b")));

    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/b/../x/c"), fs::path("/a/b")));
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/x/c"), fs::path("/a/d/../b")));
    REQUIRE_FALSE(edm::detail::pathBeginsWith(fs::path("/a/b/../x/c"), fs::path("/a/d/../b")));
  }
}

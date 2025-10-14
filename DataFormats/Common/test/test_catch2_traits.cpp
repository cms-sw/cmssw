#include <limits>
#include <string>
#include <vector>

#include "catch2/catch_all.hpp"
#include "DataFormats/Common/interface/traits.h"

TEST_CASE("edm::key_traits", "[traits]") {
  SECTION("vector key_traits") {
    using key_type = std::vector<double>::size_type;
    REQUIRE(edm::key_traits<key_type>::value == std::numeric_limits<key_type>::max());
    REQUIRE(edm::key_traits<key_type>::value == static_cast<key_type>(-1));
  }

  SECTION("string key_traits") {
    const std::string& r = edm::key_traits<std::string>::value;
    REQUIRE(r.size() == 1);
    REQUIRE(r[0] == '\a');
  }
}

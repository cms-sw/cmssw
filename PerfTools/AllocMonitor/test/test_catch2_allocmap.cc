#include "catch.hpp"

#include "PerfTools/AllocMonitor/plugins/mea_AllocMap.h"

using namespace edm::service::moduleEventAlloc;

namespace {
  void* address(int i) { return reinterpret_cast<void*>(i); }
}  // namespace

TEST_CASE("Test ema::AllocMap", "[AllocMap]") {
  SECTION("empty") {
    AllocMap map;
    CHECK(map.size() == 0);
    CHECK(map.findOffset(nullptr) == 0);
    CHECK(map.allocationSizes().empty());
  }

  SECTION("insert in order") {
    AllocMap map;
    map.insert(address(1), 1);
    CHECK(map.size() == 1);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1}));
    map.insert(address(2), 2);
    CHECK(map.size() == 2);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.findOffset(address(2)) == 1);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1, 2}));
    map.insert(address(3), 3);
    CHECK(map.size() == 3);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.findOffset(address(2)) == 1);
    CHECK(map.findOffset(address(3)) == 2);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1, 2, 3}));

    SECTION("missing find in front") { CHECK(map.findOffset(nullptr) == map.size()); }
    SECTION("missing find off end") { CHECK(map.findOffset(address(4)) == map.size()); }
    SECTION("overwrite value") {
      map.insert(address(2), 4);
      CHECK(map.allocationSizes() == std::vector<std::size_t>({1, 4, 3}));
    }
  }
  SECTION("insert in reverse order") {
    AllocMap map;
    map.insert(address(3), 3);
    CHECK(map.size() == 1);
    CHECK(map.findOffset(address(3)) == 0);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({3}));
    map.insert(address(2), 2);
    CHECK(map.size() == 2);
    CHECK(map.findOffset(address(2)) == 0);
    CHECK(map.findOffset(address(3)) == 1);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({2, 3}));
    map.insert(address(1), 1);
    CHECK(map.size() == 3);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.findOffset(address(2)) == 1);
    CHECK(map.findOffset(address(3)) == 2);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1, 2, 3}));
  }
  SECTION("insert in middle") {
    AllocMap map;
    map.insert(address(1), 1);
    CHECK(map.size() == 1);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1}));
    map.insert(address(3), 3);
    CHECK(map.size() == 2);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.findOffset(address(3)) == 1);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1, 3}));
    SECTION("missing findOffset") { CHECK(map.findOffset(address(2)) == map.size()); }
    map.insert(address(2), 2);
    CHECK(map.size() == 3);
    CHECK(map.findOffset(address(1)) == 0);
    CHECK(map.findOffset(address(2)) == 1);
    CHECK(map.findOffset(address(3)) == 2);
    CHECK(map.allocationSizes() == std::vector<std::size_t>({1, 2, 3}));
  }
}

#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"

#include "catch2/catch_all.hpp"

#include <vector>

TEST_CASE("BitonicSort", "[BitonicSort]") {
  std::vector<int> v = {6, 4, 8, 1, 12};

  SECTION("up") {
    std::vector<int> answer = v;
    std::sort(answer.begin(), answer.end(), std::less<>());
    BitonicSort<int, std::greater<int>>(up, v.begin(), v.end());
    CHECK(v == answer);
  }
  SECTION("down") {
    std::vector<int> answer = v;
    std::sort(answer.begin(), answer.end(), std::greater<>());
    BitonicSort<int, std::greater<int>>(down, v.begin(), v.end());
    CHECK(v == answer);
  }
}

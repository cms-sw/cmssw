#include "L1Trigger/L1TCalorimeter/interface/AccumulatingSort.h"

#include "catch2/catch_all.hpp"

TEST_CASE("AccumulatingSort", "[AccumulatingSort]") {
  SECTION("empty out") {
    SECTION("ascending") {
      std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      std::vector<int> out;
      const std::vector<int> answer = {1, 2, 3};
      AccumulatingSort<int, std::greater<int>> s(3);
      s.Merge(v, out);
      CHECK(answer == out);
    }
    SECTION("descending") {
      std::vector<int> v = {9, 8, 7, 6, 5, 4, 3, 2, 1};
      std::vector<int> out;
      AccumulatingSort<int, std::greater<int>> s(3);
      s.Merge(v, out);
      const std::vector<int> answer = {9, 8, 7};
      CHECK(answer == out);
    }
    SECTION("random") {
      std::vector<int> v = {7, 3, 9, 4, 6, 5, 1, 2, 8};
      std::vector<int> out;
      AccumulatingSort<int, std::greater<int>> s(3);
      s.Merge(v, out);
      const std::vector<int> answer = {7, 3, 9};
      CHECK(answer == out);
    }
    SECTION("duplicates ascending") {
      std::vector<int> v = {1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9};
      std::vector<int> out;
      const std::vector<int> answer = {1, 2, 2};
      AccumulatingSort<int, std::greater<int>> s(3);
      s.Merge(v, out);
      CHECK(answer == out);
    }
    SECTION("duplicates descending") {
      std::vector<int> v = {9, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1};
      std::vector<int> out;
      AccumulatingSort<int, std::greater<int>> s(3);
      s.Merge(v, out);
      const std::vector<int> answer = {9, 8, 8};
      CHECK(answer == out);
    }
    SECTION("mixed") {
      std::vector<int> v = {3, 4, 5, 1, 2, 0, 8, 9};
      std::vector<int> out;
      AccumulatingSort<int, std::greater<int>> s(6);
      s.Merge(v, out);
      const std::vector<int> answer = {3, 4, 5, 1, 2, 0};
      CHECK(answer == out);
    }
  }
  SECTION("out with entries") {
    SECTION("out larger") {
      std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      std::vector<int> out = {10, 11, 12, 13};
      AccumulatingSort<int, std::greater<int>> s(out.size());
      s.Merge(v, out);
      const std::vector<int> answer = {10, 11, 12, 13};
      CHECK(answer == out);
    }
    SECTION("in larger") {
      std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      std::vector<int> out = {-10, -11, -12, -13};
      AccumulatingSort<int, std::greater<int>> s(out.size());
      s.Merge(v, out);
      const std::vector<int> answer = {1, 2, 3, 4};
      CHECK(answer == out);
    }
    SECTION("mixture") {
      std::vector<int> v = {1, 4, 8, 9, 12};
      std::vector<int> out = {0, 5, 6, 10};
      AccumulatingSort<int, std::greater<int>> s(out.size());
      s.Merge(v, out);
      const std::vector<int> answer = {1, 5, 6, 10};
      CHECK(answer == out);
    }
    SECTION("interleaved") {
      std::vector<int> v = {1, 3, 5, 7, 9};
      std::vector<int> out = {2, 4, 6, 8};
      AccumulatingSort<int, std::greater<int>> s(out.size());
      s.Merge(v, out);
      const std::vector<int> answer = {2, 4, 6, 8};
      CHECK(answer == out);
    }
    SECTION("interleaved") {
      std::vector<int> v = {2, 4, 6, 8, 10};
      std::vector<int> out = {1, 2, 3, 4};
      AccumulatingSort<int, std::greater<int>> s(out.size());
      s.Merge(v, out);
      const std::vector<int> answer = {2, 4, 6, 8};
      CHECK(answer == out);
    }
  }
}

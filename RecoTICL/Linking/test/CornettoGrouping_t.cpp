#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "RecoTICL/Linking/interface/CornettoGrouping.h"

using ticl::cornetto::groups;
using ticl::cornetto::ranksAbove;
using ticl::cornetto::roots;

TEST_CASE("More energy ranks above; at equal energy the smaller index ranks above", "[CornettoGrouping]") {
  const std::vector<float> energy{5.f, 5.f, 3.f, std::nanf("")};
  REQUIRE(ranksAbove(energy, 0, 1));
  REQUIRE_FALSE(ranksAbove(energy, 1, 0));
  REQUIRE(ranksAbove(energy, 1, 2));
  REQUIRE_FALSE(ranksAbove(energy, 2, 1));
  // A NaN energy ranks above no trackster, and no trackster ranks above it.
  REQUIRE_FALSE(ranksAbove(energy, 3, 2));
  REQUIRE_FALSE(ranksAbove(energy, 2, 3));
}

TEST_CASE("Pointer jumping gives the root of every node", "[CornettoGrouping]") {
  // Two trees: 3 -> 2 -> 1 -> 0 and 5 -> 4.
  REQUIRE(roots({0, 0, 1, 2, 4, 4}) == std::vector<unsigned int>{0, 0, 0, 0, 4, 4});
  REQUIRE(roots({}).empty());
}

TEST_CASE("A node that is not compatible with its root is a group of its own", "[CornettoGrouping]") {
  // A(0) > B(1) > C(2): B and C have the root A; C is compatible with B but not with A.
  const auto root = roots({0, 0, 1});
  const auto out = groups(root, [](unsigned int r, unsigned int i) { return !(r == 0 && i == 2); });
  REQUIRE(out == std::vector<std::vector<unsigned int>>{{0, 1}, {2}});
}

TEST_CASE("The groups are a partition, in the order of their smallest member", "[CornettoGrouping]") {
  std::mt19937 gen(4242);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  for (int trial = 0; trial < 200; ++trial) {
    // A random forest: the parent of a node is a node with a smaller index (it ranks above) or the node itself.
    const unsigned int n = 1 + trial % 60;
    std::vector<unsigned int> parents(n);
    for (unsigned int i = 0; i < n; ++i)
      parents[i] = (i == 0 || u01(gen) < 0.3f) ? i : static_cast<unsigned int>(u01(gen) * i);
    std::vector<bool> keepNode(n);
    for (unsigned int i = 0; i < n; ++i)
      keepNode[i] = u01(gen) < 0.7f;
    const auto root = roots(parents);
    const auto out = groups(root, [&](unsigned int, unsigned int i) { return keepNode[i]; });
    std::vector<int> seen(n, 0);
    unsigned int previousFront = 0;
    for (std::size_t g = 0; g < out.size(); ++g) {
      REQUIRE_FALSE(out[g].empty());
      REQUIRE(std::is_sorted(out[g].begin(), out[g].end()));
      if (g > 0)
        REQUIRE(out[g].front() > previousFront);
      previousFront = out[g].front();
      for (unsigned int i : out[g]) {
        ++seen[i];
        // A member is its group head, or a node that stays with its root, which is the head.
        REQUIRE((i == out[g].front() || (keepNode[i] && root[i] == out[g].front())));
      }
    }
    REQUIRE(std::all_of(seen.begin(), seen.end(), [](int k) { return k == 1; }));
  }
}

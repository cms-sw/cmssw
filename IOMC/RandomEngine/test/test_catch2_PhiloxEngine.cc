#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "CLHEP/Random/MixMaxRng.h"
#include "IOMC/RandomEngine/interface/PhiloxEngine.h"
#include "IOMC/RandomEngine/interface/cloneEngine.h"

#include <cmath>
#include <sstream>
#include <vector>

namespace {
  std::vector<double> draw(CLHEP::HepRandomEngine& engine, int n) {
    std::vector<double> values(n);
    engine.flatArray(n, values.data());
    return values;
  }
}  // namespace

TEST_CASE("PhiloxEngine follows the Philox stream", "[PhiloxEngine]") {
  long const seeds[4] = {11, 22, 33, 44};
  edm::PhiloxEngine engine;
  engine.setSeeds(seeds, 4);
  edm::philox::EventStream const stream(edm::philox::Counter{{11, 22, 33, 44}});
  auto const block0 = stream.bits(0, 0);
  auto const block1 = stream.bits(0, 1);
  CHECK(static_cast<unsigned int>(engine) == block0[0]);
  CHECK(static_cast<unsigned int>(engine) == block0[1]);
  CHECK(static_cast<unsigned int>(engine) == block0[2]);
  CHECK(static_cast<unsigned int>(engine) == block0[3]);
  CHECK(static_cast<unsigned int>(engine) == block1[0]);
}

TEST_CASE("PhiloxEngine values equal the Philox.h conversions", "[PhiloxEngine]") {
  long const seeds[4] = {11, 22, 33, 44};
  edm::PhiloxEngine engine;
  engine.setSeeds(seeds, 4);
  edm::philox::EventStream const stream(edm::philox::Counter{{11, 22, 33, 44}});
  auto const block0 = stream.bits(0, 0);
  auto const block1 = stream.bits(0, 1);
  CHECK(engine.flat() == edm::philox::openUniformDouble(block0[0], block0[1]));
  CHECK(engine.flat() == edm::philox::openUniformDouble(block0[2], block0[3]));
  CHECK(static_cast<float>(engine) == edm::philox::openUniformFloat(block1[0]));
}

TEST_CASE("PhiloxEngine values are in (0, 1)", "[PhiloxEngine]") {
  edm::PhiloxEngine engine(1);
  for (double value : draw(engine, 100000)) {
    REQUIRE(value > 0.);
    REQUIRE(value < 1.);
  }
  for (int i = 0; i < 100000; ++i) {
    float const value = static_cast<float>(engine);
    REQUIRE(value > 0.f);
    REQUIRE(value < 1.f);
  }
}

TEST_CASE("PhiloxEngine reseed restarts the stream", "[PhiloxEngine]") {
  long const seeds[4] = {1, 2, 3, 4};
  edm::PhiloxEngine engine;
  engine.setSeeds(seeds, 4);
  auto const first = draw(engine, 7);
  engine.setSeeds(seeds, 4);
  CHECK(draw(engine, 7) == first);
  long const otherSeeds[4] = {1, 2, 3, 5};
  engine.setSeeds(otherSeeds, 4);
  CHECK(draw(engine, 7) != first);
}

TEST_CASE("PhiloxEngine state round trip", "[PhiloxEngine]") {
  long const seeds[4] = {5, 6, 7, 8};
  for (int skip : {0, 1, 2, 3, 4, 5}) {
    edm::PhiloxEngine engine;
    engine.setSeeds(seeds, 4);
    for (int i = 0; i < skip; ++i) {
      static_cast<void>(static_cast<unsigned int>(engine));
    }
    auto const state = engine.put();
    std::ostringstream text;
    engine.put(text);
    auto const expected = draw(engine, 9);

    edm::PhiloxEngine fromVector;
    REQUIRE(fromVector.get(state));
    CHECK(draw(fromVector, 9) == expected);

    edm::PhiloxEngine fromText;
    std::istringstream input(text.str());
    REQUIRE(fromText.get(input));
    CHECK(draw(fromText, 9) == expected);
  }
}

TEST_CASE("PhiloxEngine rejects a foreign state", "[PhiloxEngine]") {
  CLHEP::MixMaxRng mixMax(1);
  edm::PhiloxEngine engine;
  CHECK_FALSE(engine.get(mixMax.put()));
}

TEST_CASE("PhiloxEngine clone continues the sequence", "[PhiloxEngine]") {
  long const seeds[4] = {9, 10, 11, 12};
  edm::PhiloxEngine engine;
  engine.setSeeds(seeds, 4);
  static_cast<void>(static_cast<unsigned int>(engine));
  auto const clone = edm::cloneEngine(engine);
  REQUIRE(clone->name() == edm::PhiloxEngine::engineName());
  CHECK(draw(*clone, 9) == draw(engine, 9));
}

TEST_CASE("PhiloxEngine setSeed uses the 64 bits of the seed", "[PhiloxEngine]") {
  edm::PhiloxEngine low(5);
  edm::PhiloxEngine high(5 + (1L << 32));
  CHECK(draw(low, 4) != draw(high, 4));
}

TEST_CASE("PhiloxEngine keeps its state for an invalid input", "[PhiloxEngine]") {
  edm::PhiloxEngine engine(3);
  auto state = engine.put();
  auto const expected = draw(engine, 5);
  REQUIRE(engine.get(state));
  auto invalid = state;
  invalid[1] += 1;
  invalid[7] = 2;  // a partly used block with no previous block
  CHECK_FALSE(engine.get(invalid));
  CHECK(draw(engine, 5) == expected);
}

TEST_CASE("PhiloxEngine moments", "[PhiloxEngine]") {
  // Mean 1/2, variance 1/12 and lag-1 correlation 0, each within 5 standard deviations.
  constexpr int kCount = 1 << 20;
  edm::PhiloxEngine engine(7);
  auto const values = draw(engine, kCount);
  double sum = 0., sum2 = 0., sumLag = 0.;
  for (int i = 0; i < kCount; ++i) {
    sum += values[i];
    sum2 += values[i] * values[i];
    if (i > 0) {
      sumLag += (values[i] - 0.5) * (values[i - 1] - 0.5);
    }
  }
  double const mean = sum / kCount;
  double const variance = sum2 / kCount - mean * mean;
  CHECK(std::abs(mean - 0.5) < 5. * std::sqrt(1. / 12. / kCount));
  CHECK(std::abs(variance - 1. / 12.) < 5. * std::sqrt((1. / 80. - 1. / 144.) / kCount));
  CHECK(std::abs(sumLag / kCount * 12.) < 5. / std::sqrt(kCount));
}

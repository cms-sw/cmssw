#include "FWCore/Utilities/interface/Philox.h"

#include "catch2/catch_all.hpp"

#include <cmath>
#include <set>
#include <utility>

namespace {
  using edm::philox::Counter;
  using edm::philox::Key;

  // Known-answer vectors for philox4x32-10 from Random123 tests/kat_vectors.
  struct KnownAnswer {
    Counter counter;
    Key key;
    Counter expected;
  };
  constexpr KnownAnswer kKnownAnswers[] = {
      {{{0x00000000, 0x00000000, 0x00000000, 0x00000000}},
       {{0x00000000, 0x00000000}},
       {{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}}},
      {{{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}},
       {{0xffffffff, 0xffffffff}},
       {{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}}},
      {{{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}},
       {{0xa4093822, 0x299f31d0}},
       {{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}}},
  };

  constexpr bool knownAnswersPass() {
    for (auto const& answer : kKnownAnswers) {
      if (edm::philox::philox4x32(answer.counter, answer.key) != answer.expected) {
        return false;
      }
    }
    return true;
  }
  static_assert(knownAnswersPass(), "Philox4x32-10 must reproduce the Random123 known-answer vectors");
}  // namespace

TEST_CASE("Philox4x32-10 known answers", "[Philox]") {
  for (auto const& answer : kKnownAnswers) {
    CHECK(edm::philox::philox4x32(answer.counter, answer.key) == answer.expected);
  }
}

TEST_CASE("EventStream depends on every identity field", "[Philox]") {
  edm::philox::EventStream const reference(1234567, 42, 1, 1, 1);
  auto const referenceBits = reference.bits(0, 0);
  CHECK(edm::philox::EventStream(1234567, 42, 1, 1, 1).bits(0, 0) == referenceBits);
  CHECK(edm::philox::EventStream(1234568, 42, 1, 1, 1).bits(0, 0) != referenceBits);
  CHECK(edm::philox::EventStream(1234567, 43, 1, 1, 1).bits(0, 0) != referenceBits);
  CHECK(edm::philox::EventStream(1234567, 42, 2, 1, 1).bits(0, 0) != referenceBits);
  CHECK(edm::philox::EventStream(1234567, 42, 1, 2, 1).bits(0, 0) != referenceBits);
  CHECK(edm::philox::EventStream(1234567, 42, 1, 1, 2).bits(0, 0) != referenceBits);
  CHECK(edm::philox::EventStream(1234567, 42, 1, 1, 1 + (std::uint64_t{1} << 32)).bits(0, 0) != referenceBits);
  CHECK(reference.bits(1, 0) != referenceBits);
  CHECK(reference.bits(0, 1) != referenceBits);
}

TEST_CASE("EventStream outputs do not repeat", "[Philox]") {
  edm::philox::EventStream const stream(1, 2, 3, 4, 5);
  std::set<Counter> outputs;
  constexpr std::uint32_t kElements = 256;
  constexpr std::uint32_t kBlocks = 16;
  for (std::uint32_t element = 0; element < kElements; ++element) {
    for (std::uint32_t block = 0; block < kBlocks; ++block) {
      outputs.insert(stream.bits(element, block));
    }
  }
  CHECK(outputs.size() == kElements * kBlocks);
}

TEST_CASE("Uniform conversions", "[Philox]") {
  CHECK(edm::philox::uniformFloat(0) == 0.f);
  CHECK(edm::philox::uniformFloat(0xffffffff) == 1.f - 0x1p-24f);
  CHECK(edm::philox::uniformFloat(0xffffffff) < 1.f);
  CHECK(edm::philox::uniformDouble(0, 0) == 0.);
  CHECK(edm::philox::uniformDouble(0xffffffff, 0xffffffff) == 1. - 0x1p-53);
  CHECK(edm::philox::uniformDouble(0xffffffff, 0xffffffff) < 1.);
  CHECK(edm::philox::uniformDouble(0, 1) == 0x1p-32);
  CHECK(edm::philox::openUniformFloat(0) == 0x1p-24f);
  CHECK(edm::philox::openUniformFloat(0xffffffff) == 1.f - 0x1p-24f);
  CHECK(edm::philox::openUniformDouble(0, 0) == 0x1p-53);
  CHECK(edm::philox::openUniformDouble(0xffffffff, 0xffffffff) == 1. - 0x1p-53);
}

TEST_CASE("Moments of the uniform values", "[Philox]") {
  // Mean 1/2, variance 1/12 and lag-1 correlation 0, each within 5 standard deviations,
  // within one stream and across the streams of consecutive events.
  constexpr int kCount = 1 << 20;
  auto check = [](auto const& value) {
    double sum = 0., sum2 = 0., sumLag = 0., previous = 0.;
    for (int i = 0; i < kCount; ++i) {
      double const x = value(i);
      sum += x;
      sum2 += x * x;
      sumLag += (x - 0.5) * (previous - 0.5);
      previous = x;
    }
    double const mean = sum / kCount;
    double const variance = sum2 / kCount - mean * mean;
    double const correlation = sumLag / kCount * 12.;
    CHECK(std::abs(mean - 0.5) < 5. * std::sqrt(1. / 12. / kCount));
    CHECK(std::abs(variance - 1. / 12.) < 5. * std::sqrt((1. / 80. - 1. / 144.) / kCount));
    CHECK(std::abs(correlation) < 5. / std::sqrt(kCount));
  };
  edm::philox::EventStream const stream(1234567, 42, 1, 1, 1);
  check([&](int i) {
    auto const bits = stream.bits(0, static_cast<std::uint32_t>(i / 2));
    return edm::philox::uniformDouble(bits[2 * (i % 2)], bits[2 * (i % 2) + 1]);
  });
  check([](int i) {
    auto const bits = edm::philox::EventStream(1234567, 42, 1, 1, static_cast<std::uint64_t>(i) + 1).bits(0, 0);
    return edm::philox::uniformDouble(bits[0], bits[1]);
  });
}

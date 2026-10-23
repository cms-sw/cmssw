#include "catch2/catch_all.hpp"

#include "CondFormats/SiPixelObjects/interface/SiPixelClusterShapeLimits.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  using Limits = SiPixelClusterShapeLimits;
  constexpr int any = Limits::kAny;
  // part dx dy limits[kNLimits]
  std::vector<Limits::Entry> someEntries() { return {{0, 1, 2, std::vector<float>(Limits::kNLimits, 0.5f)}}; }
}  // namespace

TEST_CASE("SiPixelClusterShapeLimits rules", "[SiPixelClusterShapeLimits]") {
  SECTION("Phase-1 split: BPix layer 1 / everything else") {
    Limits limits;
    const auto noL1 = limits.addTable("pixelShapePhase1_noL1", someEntries());
    const auto loose = limits.addTable("pixelShapePhase1_loose", someEntries());
    limits.addRule({Limits::kBPix, 1, loose});   // BPix layer 1
    limits.addRule({Limits::kBPix, any, noL1});  // any other BPix layer
    limits.addRule({Limits::kFPix, any, noL1});  // any FPix disk
    REQUIRE_NOTHROW(limits.checkComplete());

    CHECK(limits.tableIndex(Limits::kBPix, 1) == int(loose));
    for (int layer : {2, 3, 4})
      CHECK(limits.tableIndex(Limits::kBPix, layer) == int(noL1));
    for (int disk : {1, 2, 3})
      CHECK(limits.tableIndex(Limits::kFPix, disk) == int(noL1));
  }

  SECTION("one table per BPix layer, FPix disk 1 separate") {
    Limits limits;
    for (int i = 0; i < 6; ++i)
      limits.addTable("table" + std::to_string(i), someEntries());
    // subdet, layer/disk, table id
    for (int layer : {1, 2, 3})
      limits.addRule({Limits::kBPix, layer, unsigned(layer - 1)});
    limits.addRule({Limits::kBPix, any, 3});  // catch-all: here only layer 4 can end up in this rule
    limits.addRule({Limits::kFPix, 1, 4});    // disk 1
    limits.addRule({Limits::kFPix, any, 5});  // catch-all: not disk 1
    REQUIRE_NOTHROW(limits.checkComplete());

    for (int layer : {1, 2, 3, 4})
      CHECK(limits.tableIndex(Limits::kBPix, layer) == layer - 1);
    CHECK(limits.tableIndex(Limits::kFPix, 1) == 4);
    CHECK(limits.tableIndex(Limits::kFPix, 2) == 5);
    CHECK(limits.tableIndex(Limits::kFPix, 3) == 5);
  }

  SECTION("the first matching rule wins") {
    Limits limits;
    limits.addTable("a", someEntries());
    limits.addTable("b", someEntries());
    limits.addRule({Limits::kBPix, 2, 0});  // layer 2 -> a
    limits.addRule({Limits::kBPix, 2, 1});  // layer 2 -> b: never used, layer 2 already matched above
    limits.addRule({Limits::kBPix, any, 1});
    CHECK(limits.tableIndex(Limits::kBPix, 2) == 0);
    CHECK(limits.tableIndex(Limits::kBPix, 3) == 1);
  }

  SECTION("invalid payloads are rejected") {
    Limits limits;
    limits.addTable("a", someEntries());
    CHECK_THROWS_AS(limits.addTable("bad", {{0, 1, 2, {1.f, 2.f}}}), cms::Exception);  // not kNLimits limits
    CHECK_THROWS_AS(limits.addRule({Limits::kBPix, any, 7}), cms::Exception);          // no table 7
    CHECK_THROWS_AS(limits.addRule({3, any, 0}), cms::Exception);                      // not a pixel sub-detector
    limits.addRule({Limits::kBPix, any, 0});
    CHECK_THROWS_AS(limits.addRule({Limits::kBPix, 2, 0}), cms::Exception);  // after the BPix catch-all
    CHECK_THROWS_AS(limits.checkComplete(), cms::Exception);                 // no FPix catch-all
    CHECK(limits.tableIndex(Limits::kFPix, 1) == -1);
  }
}

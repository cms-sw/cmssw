#include "catch2/catch_all.hpp"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
  // the Run 3 ideal geometry, from the XML description
  std::string configuration(const std::string& plugin, const std::string& parameters = "") {
    return R"cfg(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load('Configuration.Geometry.GeometryExtended2025Reco_cff')
process.toTest = cms.EDProducer(')cfg" +
           plugin + R"cfg(', src = cms.InputTag('hits'), name = cms.string('Hits'), doc = cms.string('the hits'))cfg" +
           parameters + R"cfg()
process.moduleToTest(process.toTest)
)cfg";
  }

  template <typename T>
  std::vector<T> column(const nanoaod::FlatTable& table, const std::string& name) {
    REQUIRE(table.columnIndex(name) >= 0);
    const auto data = table.columnData<T>(table.columnIndex(name));
    return {data.begin(), data.end()};
  }

  template <typename C>
  nanoaod::FlatTable table(const std::string& plugin, std::unique_ptr<C> hits, const std::string& parameters = "") {
    edm::test::TestProcessor::Config config{configuration(plugin, parameters)};
    const auto token = config.produces<C>("hits");
    edm::test::TestProcessor tester(config);
    const auto event = tester.test(std::make_pair(token, std::move(hits)));
    return *event.template get<nanoaod::FlatTable>();
  }

  // the first valid endcap crystal and preshower strip at z > 0
  EEDetId endcapCrystal() {
    for (int ix = EEDetId::IX_MIN; ix <= EEDetId::IX_MAX; ++ix)
      for (int iy = EEDetId::IY_MIN; iy <= EEDetId::IY_MAX; ++iy)
        if (EEDetId::validDetId(ix, iy, 1))
          return EEDetId(ix, iy, 1);
    throw cms::Exception("Test") << "no endcap crystal";
  }
  ESDetId preshowerStrip() {
    for (int ix = ESDetId::IX_MIN; ix <= ESDetId::IX_MAX; ++ix)
      for (int iy = ESDetId::IY_MIN; iy <= ESDetId::IY_MAX; ++iy)
        if (ESDetId::validDetId(1, ix, iy, 1, 1))
          return ESDetId(1, ix, iy, 1, 1);
    throw cms::Exception("Test") << "no preshower strip";
  }

  // ECAL hits: two barrel crystals mirrored in z, one at the end of the barrel, an endcap crystal, a preshower strip
  std::unique_ptr<EcalRecHitCollection> ecalHits() {
    auto hits = std::make_unique<EcalRecHitCollection>();
    hits->push_back(EcalRecHit(EBDetId(1, 1), 1.f, 2.f));
    hits->push_back(EcalRecHit(EBDetId(-1, 1), 3.f, -4.f));
    hits->push_back(EcalRecHit(EBDetId(85, 180), 5.f, 0.5f));
    hits->push_back(EcalRecHit(endcapCrystal(), 7.f, 1.f));
    hits->push_back(EcalRecHit(preshowerStrip(), 0.25f, 3.f));
    return hits;
  }
}  // namespace

TEST_CASE("CaloRecHitTableProducer: one row per hit with its position", "[CaloRecHitTableProducer]") {
  const auto out = table("EcalRecHitTableProducer", ecalHits());
  REQUIRE(out.name() == "Hits");
  REQUIRE(out.doc() == "the hits");
  REQUIRE_FALSE(out.extension());
  REQUIRE(out.nRows() == 5);
  REQUIRE(out.nColumns() == 6);
  REQUIRE(column<uint32_t>(out, "detId") == std::vector<uint32_t>{EBDetId(1, 1).rawId(),
                                                                  EBDetId(-1, 1).rawId(),
                                                                  EBDetId(85, 180).rawId(),
                                                                  endcapCrystal().rawId(),
                                                                  preshowerStrip().rawId()});
  REQUIRE(column<float>(out, "energy") == std::vector<float>{1.f, 3.f, 5.f, 7.f, 0.25f});
  REQUIRE(column<float>(out, "time") == std::vector<float>{2.f, -4.f, 0.5f, 1.f, 3.f});

  const auto x = column<float>(out, "x"), y = column<float>(out, "y"), z = column<float>(out, "z");
  for (int i = 0; i < 3; ++i)  // the front faces of the barrel crystals
    REQUIRE_THAT(std::hypot(x[i], y[i]), Catch::Matchers::WithinAbs(129.7, 0.7));
  // ieta +1 and -1 are mirrored in z, at the same phi (to the crystals' tilt, a millimetre)
  REQUIRE_THAT(z[1], Catch::Matchers::WithinAbs(-z[0], 1e-3));
  REQUIRE_THAT(std::atan2(y[1], x[1]), Catch::Matchers::WithinAbs(std::atan2(y[0], x[0]), 1e-3));
  REQUIRE(z[0] > 0.f);
  REQUIRE_THAT(z[2], Catch::Matchers::WithinAbs(268.5, 2.0));  // ieta 85: the end of the barrel
  REQUIRE_THAT(z[3], Catch::Matchers::WithinAbs(317., 3.));    // endcap
  REQUIRE_THAT(z[4], Catch::Matchers::WithinAbs(305., 3.));    // preshower
}

TEST_CASE("CaloRecHitTableProducer: HCAL", "[CaloRecHitTableProducer]") {
  auto position = [](const nanoaod::FlatTable& out) {
    REQUIRE(out.nRows() == 1);
    const float x = column<float>(out, "x")[0], y = column<float>(out, "y")[0], z = column<float>(out, "z")[0];
    return std::make_pair(std::hypot(x, y), z);
  };
  SECTION("HBHE") {
    auto hits = std::make_unique<HBHERecHitCollection>();
    hits->push_back(HBHERecHit(HcalDetId(HcalBarrel, 1, 1, 1), 1.f, 0.f, 0.f));
    const auto [r, z] = position(table("HBHERecHitTableProducer", std::move(hits)));
    REQUIRE_THAT(r, Catch::Matchers::WithinAbs(181., 2.));
  }
  SECTION("HF") {
    auto hits = std::make_unique<HFRecHitCollection>();
    hits->push_back(HFRecHit(HcalDetId(HcalForward, 30, 1, 1), 1.f, 0.f));
    const auto [r, z] = position(table("HFRecHitTableProducer", std::move(hits)));
    REQUIRE_THAT(z, Catch::Matchers::WithinAbs(1120., 10.));
  }
  SECTION("HO") {
    auto hits = std::make_unique<HORecHitCollection>();
    hits->push_back(HORecHit(HcalDetId(HcalOuter, 1, 1, 4), 1.f, 0.f));
    const auto [r, z] = position(table("HORecHitTableProducer", std::move(hits)));
    REQUIRE_THAT(r, Catch::Matchers::WithinAbs(401., 12.));
  }
}

TEST_CASE("CaloRecHitTableProducer: cut and precision", "[CaloRecHitTableProducer]") {
  const auto out =
      table("EcalRecHitTableProducer", ecalHits(), ", cut = cms.string('energy > 2'), precision = cms.int32(8)");
  REQUIRE(column<uint32_t>(out, "detId") ==
          std::vector<uint32_t>{EBDetId(-1, 1).rawId(), EBDetId(85, 180).rawId(), endcapCrystal().rawId()});
  // 8 of the 23 mantissa bits are kept
  for (const auto& name : {"energy", "time", "x", "y", "z"})
    for (float value : column<float>(out, name))
      REQUIRE((std::bit_cast<uint32_t>(value) & 0x7fff) == 0);
  REQUIRE_THAT(column<float>(out, "z")[1], Catch::Matchers::WithinAbs(268.5, 2.0));
}

TEST_CASE("CaloRecHitTableProducer: missing input", "[CaloRecHitTableProducer]") {
  auto config = [](const std::string& parameters) {
    return edm::test::TestProcessor::Config{configuration("HORecHitTableProducer", parameters)};
  };
  SECTION("fails by default") {
    edm::test::TestProcessor tester(config(""));
    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
  }
  SECTION("empty table with skipNonExistingSrc") {
    edm::test::TestProcessor tester(config(", skipNonExistingSrc = cms.bool(True)"));
    const auto event = tester.test();
    const auto& out = *event.get<nanoaod::FlatTable>();
    REQUIRE(out.nRows() == 0);
    REQUIRE(out.nColumns() == 6);
  }
}

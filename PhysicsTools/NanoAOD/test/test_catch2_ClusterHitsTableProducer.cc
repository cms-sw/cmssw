#include "catch2/catch_all.hpp"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
  std::string configuration(std::string const& plugin, std::string const& parameters = "") {
    return R"cfg(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer(')cfg" +
           plugin + R"cfg(', src = cms.InputTag('clusters'), name = cms.string('Clusters'),
    hitsName = cms.string('ClusterHits'), hitsDoc = cms.string('hits of the clusters'))cfg" +
           parameters + R"cfg()
process.moduleToTest(process.toTest)
)cfg";
  }

  template <typename T>
  std::vector<T> column(nanoaod::FlatTable const& table, std::string const& name) {
    REQUIRE(table.columnIndex(name) >= 0);
    auto const data = table.columnData<T>(table.columnIndex(name));
    return {data.begin(), data.end()};
  }

  // Three clusters of energies 1, 2, 3: two hits, none, one hit.
  template <typename T>
  std::unique_ptr<std::vector<T>> clusters() {
    auto out = std::make_unique<std::vector<T>>(3);
    for (int i = 0; i < 3; ++i)
      (*out)[i].setEnergy(i + 1);
    (*out)[0].addHitAndFraction(DetId(11), 1.f);
    (*out)[0].addHitAndFraction(DetId(12), 0.5f);
    (*out)[2].addHitAndFraction(DetId(13), 0.25f);
    return out;
  }

  template <typename T>
  void testClusters(std::string const& plugin) {
    edm::test::TestProcessor::Config config{configuration(plugin)};
    auto const token = config.produces<std::vector<T>>("clusters");
    edm::test::TestProcessor tester(config);
    auto const event = tester.test(std::make_pair(token, clusters<T>()));

    auto const& counts = *event.template get<nanoaod::FlatTable>();
    REQUIRE(counts.name() == "Clusters");
    REQUIRE(counts.extension());
    REQUIRE(counts.nRows() == 3);
    REQUIRE(counts.nColumns() == 1);
    REQUIRE(column<uint16_t>(counts, "nClusterHits") == std::vector<uint16_t>{2, 0, 1});

    auto const& hits = *event.template get<nanoaod::FlatTable>("ClusterHitsTable");
    REQUIRE(hits.name() == "ClusterHits");
    REQUIRE_FALSE(hits.extension());
    REQUIRE(hits.doc() == "hits of the clusters");
    REQUIRE(hits.nRows() == 3);
    REQUIRE(column<uint32_t>(hits, "detId") == std::vector<uint32_t>{11, 12, 13});
    REQUIRE(column<float>(hits, "fraction") == std::vector<float>{1.f, 0.5f, 0.25f});
  }
}  // namespace

TEST_CASE("ClusterHitsTableProducer: counts on the cluster table and one row per hit", "[ClusterHitsTableProducer]") {
  SECTION("reco::CaloCluster") { testClusters<reco::CaloCluster>("CaloClusterHitsTableProducer"); }
  SECTION("reco::PFCluster") { testClusters<reco::PFCluster>("PFClusterHitsTableProducer"); }
  SECTION("reco::SuperCluster") { testClusters<reco::SuperCluster>("SuperClusterHitsTableProducer"); }
}

TEST_CASE("ClusterHitsTableProducer: the selection of the cluster table", "[ClusterHitsTableProducer]") {
  auto selected = [](std::string const& parameters) {
    edm::test::TestProcessor::Config config{configuration("CaloClusterHitsTableProducer", parameters)};
    auto const token = config.produces<std::vector<reco::CaloCluster>>("clusters");
    edm::test::TestProcessor tester(config);
    auto const event = tester.test(std::make_pair(token, clusters<reco::CaloCluster>()));
    auto const& hits = *event.get<nanoaod::FlatTable>("ClusterHitsTable");
    REQUIRE(column<float>(hits, "fraction").size() == hits.nRows());
    return std::make_pair(column<uint16_t>(*event.get<nanoaod::FlatTable>(), "nClusterHits"),
                          column<uint32_t>(hits, "detId"));
  };
  SECTION("cut") {
    auto const [counts, detIds] = selected(", cut = cms.string('energy > 1.5')");
    REQUIRE(counts == std::vector<uint16_t>{0, 1});
    REQUIRE(detIds == std::vector<uint32_t>{13});
  }
  SECTION("maxLen") {
    auto const [counts, detIds] = selected(", maxLen = cms.uint32(2)");
    REQUIRE(counts == std::vector<uint16_t>{2, 0});
    REQUIRE(detIds == std::vector<uint32_t>{11, 12});
  }
  SECTION("cut and maxLen") {
    auto const [counts, detIds] = selected(", cut = cms.string('energy != 2'), maxLen = cms.uint32(1)");
    REQUIRE(counts == std::vector<uint16_t>{2});
    REQUIRE(detIds == std::vector<uint32_t>{11, 12});
  }
}

TEST_CASE("ClusterHitsTableProducer: a cut on a method of the derived type", "[ClusterHitsTableProducer]") {
  edm::test::TestProcessor::Config config{
      configuration("PFClusterHitsTableProducer", ", cut = cms.string('time > 0.5')")};
  auto const token = config.produces<std::vector<reco::PFCluster>>("clusters");
  edm::test::TestProcessor tester(config);
  auto input = clusters<reco::PFCluster>();
  (*input)[2].setTime(1.f);
  auto const event = tester.test(std::make_pair(token, std::move(input)));
  REQUIRE(column<uint16_t>(*event.get<nanoaod::FlatTable>(), "nClusterHits") == std::vector<uint16_t>{1});
  REQUIRE(column<uint32_t>(*event.get<nanoaod::FlatTable>("ClusterHitsTable"), "detId") == std::vector<uint32_t>{13});
}

TEST_CASE("ClusterHitsTableProducer: a hit listed twice stays two entries", "[ClusterHitsTableProducer]") {
  edm::test::TestProcessor::Config config{configuration("SuperClusterHitsTableProducer")};
  auto const token = config.produces<std::vector<reco::SuperCluster>>("clusters");
  edm::test::TestProcessor tester(config);
  auto input = std::make_unique<std::vector<reco::SuperCluster>>(1);
  input->front().addHitAndFraction(DetId(21), 0.75f);
  input->front().addHitAndFraction(DetId(21), 0.25f);
  auto const event = tester.test(std::make_pair(token, std::move(input)));
  REQUIRE(column<uint16_t>(*event.get<nanoaod::FlatTable>(), "nClusterHits") == std::vector<uint16_t>{2});
  auto const& hits = *event.get<nanoaod::FlatTable>("ClusterHitsTable");
  REQUIRE(column<uint32_t>(hits, "detId") == std::vector<uint32_t>{21, 21});
  REQUIRE(column<float>(hits, "fraction") == std::vector<float>{0.75f, 0.25f});
}

TEST_CASE("ClusterHitsTableProducer: precision of the fraction", "[ClusterHitsTableProducer]") {
  edm::test::TestProcessor::Config config{configuration("CaloClusterHitsTableProducer", ", precision = cms.int32(4)")};
  auto const token = config.produces<std::vector<reco::CaloCluster>>("clusters");
  edm::test::TestProcessor tester(config);
  auto input = std::make_unique<std::vector<reco::CaloCluster>>(1);
  input->front().addHitAndFraction(DetId(31), 0.3f);
  auto const event = tester.test(std::make_pair(token, std::move(input)));
  auto const fraction = column<float>(*event.get<nanoaod::FlatTable>("ClusterHitsTable"), "fraction");
  REQUIRE(fraction.size() == 1);
  REQUIRE(fraction[0] != 0.3f);
  REQUIRE(fraction[0] == Catch::Approx(0.3f).epsilon(1. / 16));
}

TEST_CASE("ClusterHitsTableProducer: empty and missing clusters", "[ClusterHitsTableProducer]") {
  auto requireEmpty = [](edm::test::Event const& event) {
    REQUIRE(event.get<nanoaod::FlatTable>()->nRows() == 0);
    REQUIRE(event.get<nanoaod::FlatTable>()->columnIndex("nClusterHits") == 0);
    REQUIRE(event.get<nanoaod::FlatTable>("ClusterHitsTable")->nRows() == 0);
    REQUIRE(event.get<nanoaod::FlatTable>("ClusterHitsTable")->nColumns() == 2);
  };
  SECTION("empty collection") {
    edm::test::TestProcessor::Config config{configuration("CaloClusterHitsTableProducer")};
    auto const token = config.produces<std::vector<reco::CaloCluster>>("clusters");
    edm::test::TestProcessor tester(config);
    requireEmpty(tester.test(std::make_pair(token, std::make_unique<std::vector<reco::CaloCluster>>())));
  }
  SECTION("skipNonExistingSrc: empty tables") {
    edm::test::TestProcessor::Config config{
        configuration("CaloClusterHitsTableProducer", ", skipNonExistingSrc = cms.bool(True)")};
    edm::test::TestProcessor tester(config);
    requireEmpty(tester.test());
  }
  SECTION("skipNonExistingSrc with the clusters present") {
    edm::test::TestProcessor::Config config{
        configuration("CaloClusterHitsTableProducer", ", skipNonExistingSrc = cms.bool(True)")};
    auto const token = config.produces<std::vector<reco::CaloCluster>>("clusters");
    edm::test::TestProcessor tester(config);
    auto const event = tester.test(std::make_pair(token, clusters<reco::CaloCluster>()));
    REQUIRE(event.get<nanoaod::FlatTable>("ClusterHitsTable")->nRows() == 3);
  }
  SECTION("otherwise an error") {
    edm::test::TestProcessor::Config config{configuration("CaloClusterHitsTableProducer")};
    edm::test::TestProcessor tester(config);
    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
  }
}

TEST_CASE("ClusterHitsTableProducer: the limit of the uint16 count", "[ClusterHitsTableProducer]") {
  edm::test::TestProcessor::Config config{configuration("CaloClusterHitsTableProducer")};
  auto const token = config.produces<std::vector<reco::CaloCluster>>("clusters");
  edm::test::TestProcessor tester(config);
  auto big = std::make_unique<std::vector<reco::CaloCluster>>(1);
  for (uint32_t i = 0; i < 65535; ++i)
    big->front().addHitAndFraction(DetId(i + 1), 1.f);
  SECTION("65535 hits fit") {
    auto const event = tester.test(std::make_pair(token, std::move(big)));
    REQUIRE(column<uint16_t>(*event.get<nanoaod::FlatTable>(), "nClusterHits") == std::vector<uint16_t>{65535});
  }
  SECTION("65536 hits are an error") {
    big->front().addHitAndFraction(DetId(65536), 1.f);
    REQUIRE_THROWS_WITH(tester.test(std::make_pair(token, std::move(big))),
                        Catch::Matchers::ContainsSubstring("too large for the nClusterHits column"));
  }
}

TEST_CASE("ClusterHitsTableProducer: hitsName must differ from name", "[ClusterHitsTableProducer]") {
  edm::test::TestProcessor::Config config{R"cfg(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer('CaloClusterHitsTableProducer', src = cms.InputTag('clusters'),
    name = cms.string('Clusters'), hitsName = cms.string('Clusters'))
process.moduleToTest(process.toTest)
)cfg"};
  REQUIRE_THROWS_WITH(edm::test::TestProcessor(config), Catch::Matchers::ContainsSubstring("hitsName must differ"));
}

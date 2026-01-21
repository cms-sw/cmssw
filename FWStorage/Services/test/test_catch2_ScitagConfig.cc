#include <string>
#include <vector>

#include "FWCore/Catalog/interface/StorageURLModifier.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

void configurePluginManagerOnce() {
  static bool configured = false;
  if (!configured) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    configured = true;
  }
}

TEST_CASE("Test ScitagConfig", "[scitagconfig]") {
  configurePluginManagerOnce();

  edm::ParameterSet defaultPSet;
  std::string typeName("ScitagConfig");
  defaultPSet.addParameter("@service_type", typeName);

  SECTION("scitagconfig default parameters") {
    std::vector<edm::ParameterSet> psets = {defaultPSet};
    edm::ServiceToken token = edm::ServiceRegistry::createSet(psets);
    edm::ServiceRegistry::Operate operate(token);

    edm::Service<edm::StorageURLModifier> scitagConfig;

    std::string pfn1 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn1);
    CHECK(pfn1 == "root://example.com//store/data/file.root?scitag.flow=196664");

    std::string pfn2 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Embedded, pfn2);
    CHECK(pfn2 == "root://example.com//store/data/file.root?scitag.flow=196700");

    std::string pfn3 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::PreMixedPileup, pfn3);
    CHECK(pfn3 == "root://example.com//store/data/file.root?scitag.flow=196704");
  }

  SECTION("scitagconfig default parameters except production case") {
    edm::ParameterSet pset = defaultPSet;
    pset.addUntrackedParameter<bool>("productionCase", true);

    std::vector<edm::ParameterSet> psets = {pset};
    edm::ServiceToken token = edm::ServiceRegistry::createSet(psets);
    edm::ServiceRegistry::Operate operate(token);

    edm::Service<edm::StorageURLModifier> scitagConfig;

    std::string pfn1 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn1);
    CHECK(pfn1 == "root://example.com//store/data/file.root?scitag.flow=196656");

    std::string pfn2 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Embedded, pfn2);
    CHECK(pfn2 == "root://example.com//store/data/file.root?scitag.flow=196700");

    std::string pfn3 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::PreMixedPileup, pfn3);
    CHECK(pfn3 == "root://example.com//store/data/file.root?scitag.flow=196704");
  }

  SECTION("scitagconfig analysis case") {
    edm::ParameterSet pset = defaultPSet;
    pset.addUntrackedParameter<bool>("enable", true);
    pset.addUntrackedParameter<bool>("productionCase", false);
    edm::ParameterSet analysisPSet;
    analysisPSet.addUntrackedParameter<unsigned int>("primarySciTag", 196612);
    analysisPSet.addUntrackedParameter<unsigned int>("embeddedSciTag", 196860);
    analysisPSet.addUntrackedParameter<unsigned int>("preMixedPileupSciTag", 196705);
    pset.addUntrackedParameter<edm::ParameterSet>("analysis", analysisPSet);
    edm::ParameterSet productionPSet;
    productionPSet.addUntrackedParameter<unsigned int>("primarySciTag", 396656);
    productionPSet.addUntrackedParameter<unsigned int>("embeddedSciTag", 396700);
    productionPSet.addUntrackedParameter<unsigned int>("preMixedPileupSciTag", 396704);
    pset.addUntrackedParameter<edm::ParameterSet>("production", productionPSet);

    std::vector<edm::ParameterSet> psets = {pset};
    edm::ServiceToken token = edm::ServiceRegistry::createSet(psets);
    edm::ServiceRegistry::Operate operate(token);

    edm::Service<edm::StorageURLModifier> scitagConfig;

    std::string pfn1 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn1);
    CHECK(pfn1 == "root://example.com//store/data/file.root?scitag.flow=196612");

    std::string pfn2 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Embedded, pfn2);
    CHECK(pfn2 == "root://example.com//store/data/file.root?scitag.flow=196860");

    std::string pfn3 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::PreMixedPileup, pfn3);
    CHECK(pfn3 == "root://example.com//store/data/file.root?scitag.flow=196705");

    std::string pfn4 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Undefined, pfn4);
    CHECK(pfn4 == "root://example.com//store/data/file.root");

    std::string pfn5 = "file://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn5);
    CHECK(pfn5 == "file://example.com//store/data/file.root");

    std::string pfn6 = "file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn6);
    CHECK(pfn6 == "file.root");

    std::string pfn7 = "root://example.com//store/data/file.root?eos.app=cmst0";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn7);
    CHECK(pfn7 == "root://example.com//store/data/file.root?eos.app=cmst0&scitag.flow=196612");

    std::string pfn8 = "root://example.com//store/data/file.root?eos.app=cmst0";
    CHECK_THROWS(scitagConfig->modify(static_cast<edm::SciTagCategory>(4), pfn8));
  }

  SECTION("scitagconfig production case") {
    edm::ParameterSet pset = defaultPSet;
    pset.addUntrackedParameter<bool>("productionCase", true);
    edm::ParameterSet productionPSet;
    productionPSet.addUntrackedParameter<unsigned int>("primarySciTag", 196656);
    productionPSet.addUntrackedParameter<unsigned int>("embeddedSciTag", 196702);
    productionPSet.addUntrackedParameter<unsigned int>("preMixedPileupSciTag", 196706);
    pset.addUntrackedParameter<edm::ParameterSet>("production", productionPSet);

    std::vector<edm::ParameterSet> psets = {pset};
    edm::ServiceToken token = edm::ServiceRegistry::createSet(psets);
    edm::ServiceRegistry::Operate operate(token);

    edm::Service<edm::StorageURLModifier> scitagConfig;

    std::string pfn1 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn1);
    CHECK(pfn1 == "root://example.com//store/data/file.root?scitag.flow=196656");

    std::string pfn2 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Embedded, pfn2);
    CHECK(pfn2 == "root://example.com//store/data/file.root?scitag.flow=196702");

    std::string pfn3 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::PreMixedPileup, pfn3);
    CHECK(pfn3 == "root://example.com//store/data/file.root?scitag.flow=196706");

    std::string pfn4 = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Undefined, pfn4);
    CHECK(pfn4 == "root://example.com//store/data/file.root");

    std::string pfn5 = "file://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn5);
    CHECK(pfn5 == "file://example.com//store/data/file.root");

    std::string pfn6 = "file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn6);
    CHECK(pfn6 == "file.root");

    std::string pfn7 = "root://example.com//store/data/file.root?eos.app=cmst0";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn7);
    CHECK(pfn7 == "root://example.com//store/data/file.root?eos.app=cmst0&scitag.flow=196656");
  }

  SECTION("scitagconfig enable is false") {
    edm::ParameterSet pset = defaultPSet;
    pset.addUntrackedParameter<bool>("enable", false);

    std::vector<edm::ParameterSet> psets = {pset};
    edm::ServiceToken token = edm::ServiceRegistry::createSet(psets);
    edm::ServiceRegistry::Operate operate(token);

    edm::Service<edm::StorageURLModifier> scitagConfig;

    std::string pfn = "root://example.com//store/data/file.root";
    scitagConfig->modify(edm::SciTagCategory::Primary, pfn);
    CHECK(pfn == "root://example.com//store/data/file.root");
  }

  SECTION("scitagconfig out of range low case") {
    edm::ParameterSet pset = defaultPSet;

    edm::ParameterSet analysisPSet;
    analysisPSet.addUntrackedParameter<unsigned int>("primarySciTag", 196611);
    pset.addUntrackedParameter<edm::ParameterSet>("analysis", analysisPSet);

    std::vector<edm::ParameterSet> psets = {pset};
    CHECK_THROWS(edm::ServiceRegistry::createSet(psets));
  }

  SECTION("scitagconfig out of range high case") {
    edm::ParameterSet pset = defaultPSet;

    edm::ParameterSet analysisPSet;
    analysisPSet.addUntrackedParameter<unsigned int>("embeddedSciTag", 196861);
    pset.addUntrackedParameter<edm::ParameterSet>("analysis", analysisPSet);

    std::vector<edm::ParameterSet> psets = {pset};
    CHECK_THROWS(edm::ServiceRegistry::createSet(psets));
  }
}

#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenarioProbabilityRcd.h"

class SiPixelBadFEDChannelSimulationSanityChecker : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelBadFEDChannelSimulationSanityChecker(edm::ParameterSet const& p);
  ~SiPixelBadFEDChannelSimulationSanityChecker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------
  const edm::ESGetToken<SiPixelFEDChannelContainer, SiPixelStatusScenariosRcd> siPixelBadFEDChToken_;
  const edm::ESGetToken<SiPixelQualityProbabilities, SiPixelStatusScenarioProbabilityRcd> siPixelQPToken_;
  const bool printdebug_;
};

SiPixelBadFEDChannelSimulationSanityChecker::SiPixelBadFEDChannelSimulationSanityChecker(edm::ParameterSet const& p)
    : siPixelBadFEDChToken_(esConsumes()),
      siPixelQPToken_(esConsumes()),
      printdebug_(p.getUntrackedParameter<bool>("printDebug", true)) {
  edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker")
      << "SiPixelBadFEDChannelSimulationSanityChecker" << std::endl;
}

SiPixelBadFEDChannelSimulationSanityChecker::~SiPixelBadFEDChannelSimulationSanityChecker() {
  edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker")
      << "~SiPixelBadFEDChannelSimulationSanityChecker " << std::endl;
}

void SiPixelBadFEDChannelSimulationSanityChecker::analyze(const edm::Event& e, const edm::EventSetup& context) {
  edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker")
      << "### SiPixelBadFEDChannelSimulationSanityChecker::analyze  ###" << std::endl;
  edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker") << " ---EVENT NUMBER " << e.id().event() << std::endl;

  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiPixelStatusScenariosRcd"));
  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker") << "Record \"SiPixelStatusScenariosRcd"
                                                                << "\" does not exist " << std::endl;
  }

  //this part gets the handle of the event source and the record (i.e. the Database)
  const SiPixelFEDChannelContainer* quality_map = &context.getData(siPixelBadFEDChToken_);

  edm::eventsetup::EventSetupRecordKey recordKey2(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiPixelStatusScenarioProbabilityRcd"));

  if (recordKey2.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogWarning("SiPixelQualityProbabilitiesTestReader") << "Record \"SiPixelStatusScenarioProbabilityRcd"
                                                             << "\" does not exist " << std::endl;
  }

  //this part gets the handle of the event source and the record (i.e. the Database)
  const SiPixelQualityProbabilities* myProbabilities = &context.getData(siPixelQPToken_);

  const auto& m_probabilities = myProbabilities->getProbability_Map();
  const auto& allScenarios = quality_map->getScenarioList();

  std::unordered_set<std::string> scenariosInProb;

  // collect all scenarios appearing with non-zero probability
  for (const auto& [PUbin, scenarioMap] : m_probabilities) {
    for (const auto& [scenario, probability] : scenarioMap) {
      if (probability != 0) {
        scenariosInProb.insert(scenario);
      }
    }
  }

  // build lookup set for scenarios present in quality map
  std::unordered_set<std::string> scenarioSet(allScenarios.begin(), allScenarios.end());

  std::vector<std::string> notFound;
  notFound.reserve(scenariosInProb.size());

  for (const auto& scenario : scenariosInProb) {
    if (!scenarioSet.count(scenario)) {
      notFound.push_back(scenario);
    }
  }

  if (!notFound.empty()) {
    for (const auto& entry : notFound) {
      edm::LogWarning("SiPixelBadFEDChannelSimulationSanityChecker")
          << "Pretty worrying! the scenario: " << entry << "  is not found in the map!!" << std::endl;

      if (printdebug_) {
        edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker") << " This scenario is found in: " << std::endl;

        for (const auto& [PUbin, scenarioMap] : m_probabilities) {
          for (const auto& [scenario, probability] : scenarioMap) {
            if (scenario == entry) {
              edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker")
                  << " - PU bin " << PUbin << " with probability: " << probability << std::endl;
            }
          }
        }
      }

      edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker")
          << "==============================================" << std::endl;
    }

    edm::LogWarning("SiPixelBadFEDChannelSimulationSanityChecker")
        << " ====> A total of " << notFound.size() << " scenarios are not found in the map!" << std::endl;

  } else {
    edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker")
        << "=================================================================================" << std::endl;

    edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker")
        << " All scenarios in probability record are found in the scenario map, (all is good)!" << std::endl;

    edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker")
        << "=================================================================================" << std::endl;
  }
}

void SiPixelBadFEDChannelSimulationSanityChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Tries sanity of Pixel Stuck TBM simulation");
  desc.addUntracked<bool>("printDebug", true);
  descriptions.add("SiPixelBadFEDChannelSimulationSanityChecker", desc);
}

DEFINE_FWK_MODULE(SiPixelBadFEDChannelSimulationSanityChecker);

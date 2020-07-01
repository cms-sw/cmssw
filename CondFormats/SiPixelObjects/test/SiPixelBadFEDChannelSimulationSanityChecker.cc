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
  const bool printdebug_;
};

SiPixelBadFEDChannelSimulationSanityChecker::SiPixelBadFEDChannelSimulationSanityChecker(edm::ParameterSet const& p)
    : printdebug_(p.getUntrackedParameter<bool>("printDebug", true)) {
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
  edm::ESHandle<SiPixelFEDChannelContainer> qualityCollectionHandle;
  context.get<SiPixelStatusScenariosRcd>().get(qualityCollectionHandle);
  const SiPixelFEDChannelContainer* quality_map = qualityCollectionHandle.product();

  edm::eventsetup::EventSetupRecordKey recordKey2(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiPixelStatusScenarioProbabilityRcd>"));

  if (recordKey2.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogWarning("SiPixelQualityProbabilitiesTestReader") << "Record \"SiPixelStatusScenarioProbabilityRcd>"
                                                             << "\" does not exist " << std::endl;
  }

  //this part gets the handle of the event source and the record (i.e. the Database)
  edm::ESHandle<SiPixelQualityProbabilities> scenarioProbabilityHandle;
  context.get<SiPixelStatusScenarioProbabilityRcd>().get(scenarioProbabilityHandle);
  const SiPixelQualityProbabilities* myProbabilities = scenarioProbabilityHandle.product();

  const SiPixelFEDChannelContainer::SiPixelBadFEDChannelsScenarioMap& m_qualities = quality_map->getScenarioMap();
  SiPixelQualityProbabilities::probabilityMap m_probabilities = myProbabilities->getProbability_Map();

  std::vector<std::string> allScenarios = quality_map->getScenarioList();
  std::vector<std::string> allScenariosInProb;

  for (auto it = m_probabilities.begin(); it != m_probabilities.end(); ++it) {
    //int PUbin = it->first;
    for (const auto& entry : it->second) {
      auto scenario = entry.first;
      auto probability = entry.second;
      if (probability != 0) {
        if (std::find(allScenariosInProb.begin(), allScenariosInProb.end(), scenario) == allScenariosInProb.end()) {
          allScenariosInProb.push_back(scenario);
        }

        // if(m_qualities.find(scenario) == m_qualities.end()){
        //   edm::LogWarning("SiPixelBadFEDChannelSimulationSanityChecker") <<"Pretty worrying! the scenario: " << scenario << " (prob:" << probability << ") is not found in the map!!"<<std::endl;
        // } else {
        //   edm::LogInfo("SiPixelBadFEDChannelSimulationSanityChecker") << "scenario: "<< scenario << " is in the map! (all is good)"<< std::endl;
        // }

      }  // if prob!=0
    }    // loop on the scenarios for that PU bin
  }      // loop on PU bins

  std::vector<std::string> notFound;
  std::copy_if(allScenariosInProb.begin(),
               allScenariosInProb.end(),
               std::back_inserter(notFound),
               [&allScenarios](const std::string& arg) {
                 return (std::find(allScenarios.begin(), allScenarios.end(), arg) == allScenarios.end());
               });

  if (!notFound.empty()) {
    for (const auto& entry : notFound) {
      edm::LogWarning("SiPixelBadFEDChannelSimulationSanityChecker")
          << "Pretty worrying! the scenario: " << entry << "  is not found in the map!!" << std::endl;

      if (printdebug_) {
        edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker") << " This scenario is found in: " << std::endl;
        for (auto it = m_probabilities.begin(); it != m_probabilities.end(); ++it) {
          int PUbin = it->first;

          for (const auto& pair : it->second) {
            if (pair.first == entry) {
              edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker")
                  << " - PU bin " << PUbin << " with probability: " << pair.second << std::endl;
            }  // if the scenario matches
          }    // loop on scenarios
        }      // loop on PU
      }        // if printdebug
      edm::LogVerbatim("SiPixelBadFEDChannelSimulationSanityChecker")
          << "==============================================" << std::endl;
    }  // loop on scenarios not found

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

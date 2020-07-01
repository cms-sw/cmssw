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
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"

class SiPixelFEDChannelContainerTestReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelFEDChannelContainerTestReader(edm::ParameterSet const& p);
  ~SiPixelFEDChannelContainerTestReader() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------
  const bool printdebug_;
  const std::string formatedOutput_;
};

SiPixelFEDChannelContainerTestReader::SiPixelFEDChannelContainerTestReader(edm::ParameterSet const& p)
    : printdebug_(p.getUntrackedParameter<bool>("printDebug", true)),
      formatedOutput_(p.getUntrackedParameter<std::string>("outputFile", "")) {
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "SiPixelFEDChannelContainerTestReader" << std::endl;
}

SiPixelFEDChannelContainerTestReader::~SiPixelFEDChannelContainerTestReader() {
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "~SiPixelFEDChannelContainerTestReader " << std::endl;
}

void SiPixelFEDChannelContainerTestReader::analyze(const edm::Event& e, const edm::EventSetup& context) {
  edm::LogInfo("SiPixelFEDChannelContainerTestReader")
      << "### SiPixelFEDChannelContainerTestReader::analyze  ###" << std::endl;
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << " ---EVENT NUMBER " << e.id().event() << std::endl;

  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiPixelStatusScenariosRcd"));

  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "Record \"SiPixelStatusScenariosRcd"
                                                         << "\" does not exist " << std::endl;
  }

  //this part gets the handle of the event source and the record (i.e. the Database)
  edm::ESHandle<SiPixelFEDChannelContainer> qualityCollectionHandle;
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "got eshandle" << std::endl;

  context.get<SiPixelStatusScenariosRcd>().get(qualityCollectionHandle);
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "got context" << std::endl;

  const SiPixelFEDChannelContainer* quality_map = qualityCollectionHandle.product();
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "got SiPixelFEDChannelContainer* " << std::endl;
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "print  pointer address : " << quality_map << std::endl;
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "Size " << quality_map->size() << std::endl;
  edm::LogInfo("SiPixelFEDChannelContainerTestReader") << "Content of myQuality_Map " << std::endl;
  // use built-in method in the CondFormat to print the content
  if (printdebug_) {
    quality_map->printAll();
  }

  FILE* pFile = nullptr;
  if (!formatedOutput_.empty())
    pFile = fopen(formatedOutput_.c_str(), "w");
  if (pFile) {
    fprintf(pFile, "SiPixelFEDChannelContainer::printAll() \n");
    fprintf(pFile,
            " ========================================================================================================="
            "========== \n");

    SiPixelFEDChannelContainer::SiPixelBadFEDChannelsScenarioMap m_qualities = quality_map->getScenarioMap();

    for (auto it = m_qualities.begin(); it != m_qualities.end(); ++it) {
      fprintf(pFile,
              " ======================================================================================================="
              "============ \n");
      fprintf(pFile, "run : %s \n ", (it->first).c_str());
      for (const auto& thePixelFEDChannel : it->second) {
        DetId detId = thePixelFEDChannel.first;
        fprintf(pFile, "DetId : %i \n", detId.rawId());
        for (const auto& entry : thePixelFEDChannel.second) {
          //unsigned int fed, link, roc_first, roc_last;
          fprintf(pFile,
                  "fed : %i | link : %2i | roc_first : %2i | roc_last: %2i \n",
                  entry.fed,
                  entry.link,
                  entry.roc_first,
                  entry.roc_last);
        }
      }
    }
  }
}

void SiPixelFEDChannelContainerTestReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Reads payloads of type SiPixelFEDChannelContainer");
  desc.addUntracked<bool>("printDebug", true);
  desc.addUntracked<std::string>("outputFile", "");
  descriptions.add("SiPixelFEDChannelContainerTestReader", desc);
}

DEFINE_FWK_MODULE(SiPixelFEDChannelContainerTestReader);

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
#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenarioProbabilityRcd.h"

class SiPixelQualityProbabilitiesTestReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelQualityProbabilitiesTestReader(edm::ParameterSet const& p);
  ~SiPixelQualityProbabilitiesTestReader() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------
  const edm::ESGetToken<SiPixelQualityProbabilities, SiPixelStatusScenarioProbabilityRcd> siPixelQPToken_;
  const bool printdebug_;
  const std::string formatedOutput_;
};

SiPixelQualityProbabilitiesTestReader::SiPixelQualityProbabilitiesTestReader(edm::ParameterSet const& p)
    : siPixelQPToken_(esConsumes()),
      printdebug_(p.getUntrackedParameter<bool>("printDebug", true)),
      formatedOutput_(p.getUntrackedParameter<std::string>("outputFile", "")) {
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "SiPixelQualityProbabilitiesTestReader" << std::endl;
}

SiPixelQualityProbabilitiesTestReader::~SiPixelQualityProbabilitiesTestReader() {
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "~SiPixelQualityProbabilitiesTestReader " << std::endl;
}

void SiPixelQualityProbabilitiesTestReader::analyze(const edm::Event& e, const edm::EventSetup& context) {
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader")
      << "### SiPixelQualityProbabilitiesTestReader::analyze  ###" << std::endl;
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << " ---EVENT NUMBER " << e.id().event() << std::endl;

  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiPixelStatusScenarioProbabilityRcd"));

  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "Record \"SiPixelStatusScenarioProbabilityRcd"
                                                          << "\" does not exist " << std::endl;
  }

  //this part gets the handle of the event source and the record (i.e. the Database)
  edm::ESHandle<SiPixelQualityProbabilities> scenarioProbabilityHandle = context.getHandle(siPixelQPToken_);
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "got eshandle from context" << std::endl;

  const SiPixelQualityProbabilities* myProbabilities = scenarioProbabilityHandle.product();
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "got SiPixelQualityProbabilities* " << std::endl;
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "print  pointer address : " << myProbabilities << std::endl;

  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "Size " << myProbabilities->size() << std::endl;
  edm::LogInfo("SiPixelQualityProbabilitiesTestReader") << "Content of my Probabilities " << std::endl;
  // use built-in method in the CondFormat to print the content
  if (printdebug_) {
    myProbabilities->printAll();
  }

  FILE* pFile = nullptr;
  if (!formatedOutput_.empty())
    pFile = fopen(formatedOutput_.c_str(), "w");
  if (pFile) {
    fprintf(pFile, "SiPixelQualityProbabilities::printAll() \n");
    fprintf(pFile,
            " ========================================================================================================="
            "========== \n");

    SiPixelQualityProbabilities::probabilityMap m_probabilities = myProbabilities->getProbability_Map();

    for (auto it = m_probabilities.begin(); it != m_probabilities.end(); ++it) {
      fprintf(pFile,
              " ======================================================================================================="
              "============ \n");
      fprintf(pFile, "PU bin : %i \n ", (it->first));
      for (const auto& entry : it->second) {
        fprintf(pFile, "Quality snapshot %s, probability %f \n", (entry.first).c_str(), entry.second);
      }
    }
  }
}

void SiPixelQualityProbabilitiesTestReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Reads payloads of type SiPixelQualityProbabilities");
  desc.addUntracked<bool>("printDebug", true);
  desc.addUntracked<std::string>("outputFile", "");
  descriptions.add("SiPixelQualityProbabilitiesTestReader", desc);
}

DEFINE_FWK_MODULE(SiPixelQualityProbabilitiesTestReader);

// -*- C++ -*-
//
// Package:    CondFormats/SiPixelObjects
// Class:      SiPixelQualityProbabilitiesTestWriter
//
/**\class SiPixelQualityProbabilitiesTestWriter SiPixelQualityProbabilitiesTestWriter.cc CondFormats/SiPixelObjects/plugins/SiPixelQualityProbabilitiesTestWriter.cc
 Description: class to build the SiPixel Quality probabilities 
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 30 Nov 2018 13:22:00 GMT
//
//

// system include files
#include <memory>
#include <fstream>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//

class SiPixelQualityProbabilitiesWriteFromASCII : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelQualityProbabilitiesWriteFromASCII(const edm::ParameterSet&);
  ~SiPixelQualityProbabilitiesWriteFromASCII() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const std::string m_ProbInputs;
  const std::string m_record;
  const bool printdebug_;
  SiPixelQualityProbabilities* myProbabilities;
};

//
// constructors and destructor
//
SiPixelQualityProbabilitiesWriteFromASCII::SiPixelQualityProbabilitiesWriteFromASCII(const edm::ParameterSet& iConfig)
    : m_ProbInputs(iConfig.getParameter<std::string>("probabilities")),
      m_record(iConfig.getParameter<std::string>("record")),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {
  //now do what ever initialization is needed
  myProbabilities = new SiPixelQualityProbabilities();
}

SiPixelQualityProbabilitiesWriteFromASCII::~SiPixelQualityProbabilitiesWriteFromASCII() { delete myProbabilities; }

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelQualityProbabilitiesWriteFromASCII::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::ifstream myfile(m_ProbInputs);
  std::string line;

  SiPixelQualityProbabilities::probabilityVec myProbVector;

  if (myfile.is_open()) {
    while (getline(myfile, line)) {
      if (printdebug_) {
        edm::LogInfo("SiPixelQualityProbabilitiesWriteFromASCII") << line << std::endl;
      }
      std::istringstream iss(line);
      int pileupBinId, nEntries;
      iss >> pileupBinId >> nEntries;
      edm::LogInfo("SiPixelQualityProbabilitiesWriteFromASCII")
          << "PILEUP BIN/ENTRIES:  " << pileupBinId << " " << nEntries << std::endl;
      std::vector<std::string> ids(nEntries, "");
      std::vector<float> probs(nEntries, 0.0);
      for (int i = 0; i < nEntries; ++i) {
        iss >> ids.at(i) >> probs.at(i);
        if (printdebug_) {
          edm::LogInfo("SiPixelQualityProbabilitiesWriteFromASCII") << ids.at(i) << " " << probs.at(i) << std::endl;
        }
        auto idAndProb = std::make_pair(ids.at(i), probs.at(i));
        myProbVector.push_back(idAndProb);
      }
      if (nEntries > 0) {
        myProbabilities->setProbabilities(pileupBinId, myProbVector);
      }
      myProbVector.clear();
    }
    myfile.close();
  }

  if (printdebug_) {
    edm::LogInfo("SiPixelQualityProbabilitiesWriteFromASCII") << "Content of SiPixelQualityProbabilities " << std::endl;
    // use buil-in method in the CondFormat
    myProbabilities->printAll();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiPixelQualityProbabilitiesWriteFromASCII::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiPixelQualityProbabilitiesWriteFromASCII::endJob() {
  edm::LogInfo("SiPixelQualityProbabilitiesWriteFromASCII")
      << "Size of SiPixelQualityProbabilities object " << myProbabilities->size() << std::endl
      << std::endl;

  // Form the data here
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    cond::Time_t valid_time = poolDbService->currentTime();
    // this writes the payload to begin in current run defined in cfg
    poolDbService->writeOne(myProbabilities, valid_time, m_record);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelQualityProbabilitiesWriteFromASCII::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Writes payloads of type SiPixelQualityProbabilities");
  desc.addUntracked<bool>("printDebug", true);
  desc.add<std::string>("record", "SiPixelStatusScenarioProbabilityRcd");
  desc.add<std::string>("probabilities", "");
  descriptions.add("SiPixelQualityProbabilitiesWriteFromASCII", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelQualityProbabilitiesWriteFromASCII);

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

class SiPixelQualityProbabilitiesTestWriter : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelQualityProbabilitiesTestWriter(const edm::ParameterSet&);
  ~SiPixelQualityProbabilitiesTestWriter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const std::string m_ProbInputs;
  const std::string m_SnapshotInputs;
  const std::string m_record;
  const bool printdebug_;
  SiPixelQualityProbabilities* myProbabilities;
};

//
// constructors and destructor
//
SiPixelQualityProbabilitiesTestWriter::SiPixelQualityProbabilitiesTestWriter(const edm::ParameterSet& iConfig)
    : m_ProbInputs(iConfig.getParameter<std::string>("probabilities")),
      m_SnapshotInputs(iConfig.getParameter<std::string>("snapshots")),
      m_record(iConfig.getParameter<std::string>("record")),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {
  //now do what ever initialization is needed
  myProbabilities = new SiPixelQualityProbabilities();
}

SiPixelQualityProbabilitiesTestWriter::~SiPixelQualityProbabilitiesTestWriter() { delete myProbabilities; }

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelQualityProbabilitiesTestWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::ifstream myfile(m_ProbInputs);
  std::ifstream mysnapshots(m_SnapshotInputs);
  std::string line1, line2;
  std::map<int, std::string> snapshotIdToString;

  if (mysnapshots.is_open()) {
    while (getline(mysnapshots, line1)) {
      //edm::LogInfo("SiPixelQualityProbabilitiesTestWriter") << line1 << std::endl;
      std::istringstream iss(line1);
      int id, run, ls;
      iss >> id >> run >> ls;
      snapshotIdToString[id] = std::to_string(run) + "_" + std::to_string(ls);
    }
  }

  SiPixelQualityProbabilities::probabilityVec myProbVector;

  if (myfile.is_open()) {
    while (getline(myfile, line2)) {
      edm::LogInfo("SiPixelQualityProbabilitiesTestWriter") << line2 << std::endl;
      std::istringstream iss(line2);
      int pileupBinId, nEntries;
      iss >> pileupBinId >> nEntries;
      edm::LogInfo("SiPixelQualityProbabilitiesTestWriter")
          << "PILEUP BIN/ENTRIES:  " << pileupBinId << " " << nEntries << std::endl;
      std::vector<int> ids(nEntries, 0);
      std::vector<float> probs(nEntries, 0.0);
      for (int i = 0; i < nEntries; ++i) {
        iss >> ids.at(i) >> probs.at(i);
        //edm::LogInfo("SiPixelQualityProbabilitiesTestWriter") << ids.at(i) << " " << probs.at(i)<< std::endl;
        auto idAndProb = std::make_pair(snapshotIdToString.at(ids.at(i)), probs.at(i));
        myProbVector.push_back(idAndProb);
      }
      if (nEntries > 0)
        myProbabilities->setProbabilities(pileupBinId, myProbVector);
      myProbVector.clear();
    }
    myfile.close();
  }

  if (printdebug_) {
    edm::LogInfo("SiPixelQualityProbabilitiesTestWriter") << "Content of SiPixelQualityProbabilities " << std::endl;
    // use buil-in method in the CondFormat
    myProbabilities->printAll();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiPixelQualityProbabilitiesTestWriter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiPixelQualityProbabilitiesTestWriter::endJob() {
  edm::LogInfo("SiPixelQualityProbabilitiesTestWriter")
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
void SiPixelQualityProbabilitiesTestWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Writes payloads of type SiPixelQualityProbabilities");
  desc.addUntracked<bool>("printDebug", true);
  desc.add<std::string>("record", "SiPixelStatusScenarioProbabilityRcd");
  desc.add<std::string>("snapshots", "");
  desc.add<std::string>("probabilities", "");
  descriptions.add("SiPixelQualityProbabilitiesTestWriter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelQualityProbabilitiesTestWriter);

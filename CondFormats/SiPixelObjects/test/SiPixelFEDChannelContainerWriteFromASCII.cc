// -*- C++ -*-
//
// Package:    CondFormats/SiPixelObjects
// Class:      SiPixelFEDChannelContainerTestWriter
//
/**\class SiPixelFEDChannelContainerTestWriter SiPixelFEDChannelContainerTestWriter.cc CondFormats/SiPixelObjects/plugins/SiPixelFEDChannelContainerTestWriter.cc
 Description: class to build the SiPixelFEDChannelContainer payloads
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 27 Nov 2018 12:04:36 GMT
//
//

// system include files
#include <memory>
#include <fstream>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
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

class SiPixelFEDChannelContainerWriteFromASCII : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelFEDChannelContainerWriteFromASCII(const edm::ParameterSet&);
  ~SiPixelFEDChannelContainerWriteFromASCII() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const std::string m_record;
  const std::string m_SnapshotInputs;
  const bool printdebug_;
  const bool addDefault_;
  SiPixelFEDChannelContainer* myQualities;
};

//
// constructors and destructor
//
SiPixelFEDChannelContainerWriteFromASCII::SiPixelFEDChannelContainerWriteFromASCII(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")),
      m_SnapshotInputs(iConfig.getParameter<std::string>("snapshots")),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      addDefault_(iConfig.getUntrackedParameter<bool>("addDefault", false)) {
  //now do what ever initialization is needed
  myQualities = new SiPixelFEDChannelContainer();
}

SiPixelFEDChannelContainerWriteFromASCII::~SiPixelFEDChannelContainerWriteFromASCII() { delete myQualities; }

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelFEDChannelContainerWriteFromASCII::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::ifstream mysnapshots(m_SnapshotInputs);
  std::string line;

  std::string scenario = "";
  unsigned int thedetid(0);

  SiPixelFEDChannelContainer::SiPixelFEDChannelCollection theBadFEDChannels;

  if (mysnapshots.is_open()) {
    while (getline(mysnapshots, line)) {
      if (printdebug_) {
        edm::LogVerbatim("SiPixelFEDChannelContainerWriteFromASCII") << line << std::endl;
      }
      std::istringstream iss(line);
      unsigned int run, ls, detid, fed, link, roc_first, roc_last;
      iss >> run >> ls >> detid >> fed >> link >> roc_first >> roc_last;

      PixelFEDChannel theBadChannel{fed, link, roc_first, roc_last};

      auto newscenario = std::to_string(run) + "_" + std::to_string(ls);
      if (newscenario != scenario) {
        edm::LogVerbatim("SiPixelFEDChannelContainerWriteFromASCII") << "================================" << std::endl;
        edm::LogVerbatim("SiPixelFEDChannelContainerWriteFromASCII")
            << "found a new scenario: " << newscenario << std::endl;
        if (!scenario.empty()) {
          edm::LogVerbatim("SiPixelFEDChannelContainerWriteFromASCII")
              << "size of the fed channel vector: " << theBadFEDChannels.size() << std::endl;
          edm::LogVerbatim("SiPixelFEDChannelContainerWriteFromASCII")
              << "================================" << std::endl;
          myQualities->setScenario(scenario, theBadFEDChannels);
          theBadFEDChannels.clear();
        }
        scenario = newscenario;
      }

      if (detid != thedetid) {
        if (printdebug_) {
          edm::LogVerbatim("SiPixelFEDChannelContainerWriteFromASCII") << "found a new detid!" << detid << std::endl;
        }
        thedetid = detid;
      }
      theBadFEDChannels[thedetid].push_back(theBadChannel);
    }
  }

  myQualities->setScenario(scenario, theBadFEDChannels);

  if (printdebug_) {
    edm::LogInfo("SiPixelFEDChannelContainerWriteFromASCII") << "Content of SiPixelFEDChannelContainer " << std::endl;

    // use buil-in method in the CondFormat
    myQualities->printAll();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiPixelFEDChannelContainerWriteFromASCII::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiPixelFEDChannelContainerWriteFromASCII::endJob() {
  // adds an empty payload with name "default" => no channels are masked
  if (addDefault_) {
    SiPixelFEDChannelContainer::SiPixelFEDChannelCollection theBadFEDChannels;
    myQualities->setScenario("default", theBadFEDChannels);
  }

  edm::LogInfo("SiPixelFEDChannelContainerWriteFromASCII")
      << "Size of SiPixelFEDChannelContainer object " << myQualities->size() << std::endl
      << std::endl;

  // Form the data here
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    cond::Time_t valid_time = poolDbService->currentTime();
    // this writes the payload to begin in current run defined in cfg
    poolDbService->writeOne(myQualities, valid_time, m_record);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelFEDChannelContainerWriteFromASCII::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Writes payloads of type SiPixelFEDChannelContainer from input ASCII files");
  desc.addUntracked<bool>("printDebug", true);
  desc.addUntracked<bool>("addDefault", true);
  desc.add<std::string>("snapshots", "");
  desc.add<std::string>("record", "SiPixelStatusScenariosRcd");
  descriptions.add("SiPixelFEDChannelContainerWriteFromASCII", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFEDChannelContainerWriteFromASCII);

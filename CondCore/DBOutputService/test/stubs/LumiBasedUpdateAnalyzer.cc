#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "LumiBasedUpdateAnalyzer.h"
#include <iostream>

LumiBasedUpdateAnalyzer::LumiBasedUpdateAnalyzer(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")), m_ret(-2) {}

LumiBasedUpdateAnalyzer::~LumiBasedUpdateAnalyzer() {}

void LumiBasedUpdateAnalyzer::beginJob() {
  edm::Service<cond::service::OnlineDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    return;
  }
  mydbservice->lockRecords();
}

void LumiBasedUpdateAnalyzer::endJob() {
  edm::Service<cond::service::OnlineDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    mydbservice->releaseLocks();
  }
}

void LumiBasedUpdateAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                                                   const edm::EventSetup& context) {
  edm::Service<cond::service::OnlineDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    return;
  }
  mydbservice->logger().start();
  ::sleep(2);
  unsigned int irun = lumiSeg.getRun().run();
  unsigned int lumiId = lumiSeg.luminosityBlock();
  std::string tag = mydbservice->tag(m_record);
  std::cout << "tag " << tag << std::endl;
  std::cout << "run " << irun << std::endl;
  mydbservice->logger().logDebug() << "Tag: " << tag << " Run: " << irun;
  BeamSpotObjects mybeamspot;
  mybeamspot.SetPosition(0.053, 0.1, 0.13);
  mybeamspot.SetSigmaZ(3.8);
  mybeamspot.SetType(int(lumiId));
  std::cout << mybeamspot.GetBeamType() << std::endl;
  mydbservice->logger().logDebug() << "BeamType: " << mybeamspot.GetBeamType();
  m_ret = 0;
  try {
    mydbservice->writeForNextLumisection(&mybeamspot, m_record);
  } catch (const std::exception& e) {
    std::cout << "Error:" << e.what() << std::endl;
    mydbservice->logger().logError() << e.what();
    m_ret = -1;
  }
}

void LumiBasedUpdateAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) {
  edm::Service<cond::service::OnlineDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    mydbservice->logger().logInfo() << "EndLuminosityBlock";
    mydbservice->logger().end(m_ret);
  }
}

void LumiBasedUpdateAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::cout << "LumiBasedUpdateAnalyzer::analyze " << std::endl;
}

DEFINE_FWK_MODULE(LumiBasedUpdateAnalyzer);

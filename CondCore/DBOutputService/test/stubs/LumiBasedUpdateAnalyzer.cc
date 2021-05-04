#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "LumiBasedUpdateAnalyzer.h"
#include <iostream>

LumiBasedUpdateAnalyzer::LumiBasedUpdateAnalyzer(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")) {}
LumiBasedUpdateAnalyzer::~LumiBasedUpdateAnalyzer() {
  std::cout << "LumiBasedUpdateAnalyzer::~LumiBasedUpdateAnalyzer" << std::endl;
}
void LumiBasedUpdateAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::cout << "LumiBasedUpdateAnalyzer::analyze " << std::endl;
  edm::Service<cond::service::OnlineDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    std::cout << "Service is unavailable" << std::endl;
    return;
  }
  mydbservice->logger().start();
  if (!m_tagLocks) {
    mydbservice->lockRecords();
    m_tagLocks = true;
  }
  ::sleep(2);
  //unsigned int irun = evt.id().run();
  unsigned int irun = evt.getLuminosityBlock().run();
  unsigned int lumiId = evt.getLuminosityBlock().luminosityBlock();
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
  int ret = 0;
  try {
    mydbservice->writeForNextLumisection(&mybeamspot, m_record);
  } catch (const std::exception& e) {
    std::cout << "Error:" << e.what() << std::endl;
    mydbservice->logger().logError() << e.what();
    ret = -1;
  }
  mydbservice->logger().end(ret);
}
void LumiBasedUpdateAnalyzer::endJob() {
  if (m_tagLocks) {
    edm::Service<cond::service::OnlineDBOutputService> mydbservice;
    mydbservice->releaseLocks();
  }
}
DEFINE_FWK_MODULE(LumiBasedUpdateAnalyzer);

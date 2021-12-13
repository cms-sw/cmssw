#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <iostream>
#include <string>
#include <chrono>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

// class declaration
class LumiBasedUpdateAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit LumiBasedUpdateAnalyzer(const edm::ParameterSet& iConfig);
  virtual ~LumiBasedUpdateAnalyzer();
  virtual void beginJob();
  virtual void endJob();
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  std::string m_record;
  unsigned int m_iovSize;
  std::string m_lumiFile;
  cond::Time_t m_lastLumi;
  unsigned int m_nLumi;
  int m_ret;
};

LumiBasedUpdateAnalyzer::LumiBasedUpdateAnalyzer(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getUntrackedParameter<std::string>("record")),
      m_iovSize(iConfig.getUntrackedParameter<unsigned int>("iovSize")),
      m_lumiFile(iConfig.getUntrackedParameter<std::string>("lastLumiFile")),
      m_nLumi(0),
      m_ret(-2) {}

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
                                                   const edm::EventSetup& context) {}

void LumiBasedUpdateAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) {}

void LumiBasedUpdateAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  edm::Service<cond::service::OnlineDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    return;
  }
  auto& lumiBlock = evt.getLuminosityBlock();
  unsigned int irun = lumiBlock.getRun().run();
  unsigned int lumiId = lumiBlock.luminosityBlock();
  cond::Time_t currentLumi = cond::time::lumiTime(irun, lumiId);
  auto& rec = mydbservice->lookUpRecord(m_record);
  m_ret = -1;
  mydbservice->logger().start();
  mydbservice->logger().logDebug() << "Transaction id for time " << currentLumi << " : "
                                   << cond::time::transactionIdForLumiTime(currentLumi, rec.m_refreshTime, "");
  if (currentLumi != m_lastLumi) {
    m_nLumi++;
    std::ofstream lastLumiFile(m_lumiFile, std::ofstream::trunc);
    lastLumiFile << currentLumi;
    lastLumiFile.close();
    m_lastLumi = currentLumi;
    if (m_nLumi == 3) {
      std::string tag = mydbservice->tag(m_record);
      mydbservice->logger().logDebug() << "Tag: " << tag;
      BeamSpotObjects mybeamspot;
      mybeamspot.setPosition(0.053, 0.1, 0.13);
      mybeamspot.setSigmaZ(3.8);
      mybeamspot.setType(int(lumiId));
      mydbservice->logger().logDebug() << "BeamType: " << mybeamspot.beamType();
      try {
        auto iov = mydbservice->writeIOVForNextLumisection(mybeamspot, m_record);
        if (iov) {
          auto utime = cond::time::unpack(iov);
          mydbservice->logger().logDebug() << " Run: " << irun << " Lumi: " << lumiId << " IOV lumi: " << utime.second;
          m_ret = 0;
        }
      } catch (const std::exception& e) {
        mydbservice->logger().logError() << e.what();
        m_ret = 1;
      }
      m_nLumi = 0;
    } else {
      mydbservice->logger().logDebug() << "Skipping lumisection " << lumiId;
    }
  }
  mydbservice->logger().end(m_ret);
}

DEFINE_FWK_MODULE(LumiBasedUpdateAnalyzer);

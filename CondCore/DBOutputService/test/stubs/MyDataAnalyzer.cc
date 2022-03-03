#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondFormats/Calibration/interface/Pedestals.h"

#include <string>
#include <cstdlib>
#include <iostream>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

// class decleration
class MyDataAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit MyDataAnalyzer(const edm::ParameterSet& iConfig);
  virtual ~MyDataAnalyzer();
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void beginJob();
  virtual void endJob();

private:
  std::string m_record;
  bool m_LoggingOn;
};

MyDataAnalyzer::MyDataAnalyzer(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")), m_LoggingOn(false) {
  m_LoggingOn = iConfig.getUntrackedParameter<bool>("loggingOn");
  std::cout << "MyDataAnalyzer::MyDataAnalyzer" << std::endl;
}

MyDataAnalyzer::~MyDataAnalyzer() { std::cout << "MyDataAnalyzer::~MyDataAnalyzer" << std::endl; }

void MyDataAnalyzer::beginJob() {
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    return;
  }
  mydbservice->logger().start();
  ;
}

void MyDataAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::cout << "MyDataAnalyzer::analyze " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    std::cout << "Service is unavailable" << std::endl;
    return;
  }
  try {
    std::string tag = mydbservice->tag(m_record);
    Pedestals myped;
    for (int ichannel = 1; ichannel <= 5; ++ichannel) {
      Pedestals::Item item;
      item.m_mean = 1.11 * ichannel;
      item.m_variance = 1.12 * ichannel;
      myped.m_pedestals.push_back(item);
    }
    auto t = mydbservice->currentTime();
    mydbservice->logger().logDebug() << "Writing new payload";
    mydbservice->writeOneIOV(myped, t, m_record);
    mydbservice->logger().logDebug() << "Written iov with since: " << t;
    std::cout << "done" << std::endl;
  } catch (const cond::Exception& er) {
    throw cms::Exception("DBOutputServiceUnitTestFailure", "failed MyDataAnalyzer", er);
  } catch (const cms::Exception& er) {
    throw cms::Exception("DBOutputServiceUnitTestFailure", "failed MyDataAnalyzer", er);
  }
}

void MyDataAnalyzer::endJob() {
  std::cout << "MyDataAnalyzer::endJob " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    std::cout << "Service is unavailable" << std::endl;
    return;
  }
  mydbservice->logger().end(0);
  mydbservice->logger().saveOnFile();
}

DEFINE_FWK_MODULE(MyDataAnalyzer);

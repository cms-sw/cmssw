#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Calibration/interface/Pedestals.h"

#include <iostream>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

// class decleration
class IOVPayloadAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit IOVPayloadAnalyzer(const edm::ParameterSet& iConfig);
  virtual ~IOVPayloadAnalyzer();
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  std::string m_record;
};

IOVPayloadAnalyzer::IOVPayloadAnalyzer(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")) {
  std::cout << "IOVPayloadAnalyzer::IOVPayloadAnalyzer" << std::endl;
}

IOVPayloadAnalyzer::~IOVPayloadAnalyzer() { std::cout << "IOVPayloadAnalyzer::~IOVPayloadAnalyzer" << std::endl; }

void IOVPayloadAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::cout << "IOVPayloadAnalyzer::analyze " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    std::cout << "Service is unavailable" << std::endl;
    return;
  }
  unsigned int irun = evt.id().run();
  std::string tag = mydbservice->tag(m_record);
  std::cout << "tag " << tag << std::endl;
  std::cout << "run " << irun << std::endl;
  Pedestals myped;
  for (int ichannel = 1; ichannel <= 5; ++ichannel) {
    Pedestals::Item item;
    item.m_mean = 1.11 * ichannel + irun;
    item.m_variance = 1.12 * ichannel + irun;
    myped.m_pedestals.push_back(item);
  }
  std::cout << myped.m_pedestals[1].m_mean << std::endl;

  std::cout << "currentTime " << mydbservice->currentTime() << std::endl;
  if (mydbservice->isNewTagRequest(m_record)) {
    mydbservice->createOneIOV(myped, mydbservice->currentTime(), m_record);
  } else {
    mydbservice->appendOneIOV(myped, mydbservice->currentTime(), m_record);
  }
  mydbservice->startTransaction();
  auto iov = mydbservice->currentTime() + 100;
  auto hash = mydbservice->writeOneIOV(myped, iov, m_record);
  mydbservice->commitTransaction();
  cond::TagInfo_t tinfo;
  mydbservice->tagInfo(m_record, tinfo);
  if (tinfo.lastInterval.payloadId == hash && tinfo.lastInterval.since == iov) {
    std::cout << "Last IOV stored is the expected one..." << std::endl;
  } else {
    std::cout << "Last IOV hash = " << tinfo.lastInterval.payloadId << " (expected: " << hash
              << ") since = " << tinfo.lastInterval.since << " (expected: " << iov << ")" << std::endl;
  }
  mydbservice->eraseSinceTime(hash, iov, m_record);
}

void IOVPayloadAnalyzer::endJob() { std::cout << "End of job..." << std::endl; }

DEFINE_FWK_MODULE(IOVPayloadAnalyzer);

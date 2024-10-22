#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Calibration/interface/BlobComplex.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include <iostream>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

// class decleration
class writeBlobComplex : public edm::one::EDAnalyzer<> {
public:
  explicit writeBlobComplex(const edm::ParameterSet& iConfig);
  ~writeBlobComplex();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

private:
  std::string m_RecordName;
};

writeBlobComplex::writeBlobComplex(const edm::ParameterSet& iConfig) : m_RecordName("BlobComplexRcd") {}

writeBlobComplex::~writeBlobComplex() { std::cout << "writeBlobComplex::writeBlobComplex" << std::endl; }

void writeBlobComplex::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::cout << "writeBlobComplex::analyze " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    std::cout << "db service unavailable" << std::endl;
    return;
  }
  try {
    BlobComplex me;
    unsigned int serial = 123;
    me.fill(serial);
    std::cout << "writeBlobComplex::about to write " << std::endl;
    mydbservice->writeOneIOV(me, mydbservice->currentTime(), m_RecordName);
  } catch (const std::exception& er) {
    std::cout << "caught std::exception " << er.what() << std::endl;
    throw er;
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeBlobComplex);

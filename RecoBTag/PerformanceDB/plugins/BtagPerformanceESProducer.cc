#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <iostream>
#include <memory>
#include <string>

#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

using namespace edm;

class BtagPerformanceESProducer : public edm::ESProducer {
public:
  BtagPerformanceESProducer(const edm::ParameterSet& p);
  ~BtagPerformanceESProducer() override;
  std::unique_ptr<BtagPerformance> produce(const BTagPerformanceRecord&);

private:
  edm::ESGetToken<PerformancePayload, PerformancePayloadRecord> payloadToken_;
  edm::ESGetToken<PerformanceWorkingPoint, PerformanceWPRecord> workingPointToken_;
};

BtagPerformanceESProducer::BtagPerformanceESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  auto mypl = p.getParameter<std::string>("PayloadName");
  auto mywp = p.getParameter<std::string>("WorkingPointName");

  auto c = setWhatProduced(this, myname);
  payloadToken_ = c.consumes(edm::ESInputTag("", mypl));
  workingPointToken_ = c.consumes(edm::ESInputTag("", mywp));
}

BtagPerformanceESProducer::~BtagPerformanceESProducer() {}

std::unique_ptr<BtagPerformance> BtagPerformanceESProducer::produce(const BTagPerformanceRecord& iRecord) {
  auto const& pl = iRecord.get(payloadToken_);
  auto const& wp = iRecord.get(workingPointToken_);
  return std::make_unique<BtagPerformance>(pl, wp);
}

DEFINE_FWK_EVENTSETUP_MODULE(BtagPerformanceESProducer);

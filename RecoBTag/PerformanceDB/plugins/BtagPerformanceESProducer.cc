#include "RecoBTag/PerformanceDB/plugins/BtagPerformanceESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <iostream>
#include <memory>
#include <string>

using namespace edm;

BtagPerformanceESProducer::BtagPerformanceESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  mypl = p.getParameter<std::string>("PayloadName");
  mywp = p.getParameter<std::string>("WorkingPointName");

  pset_ = p;
  setWhatProduced(this, myname);
}

BtagPerformanceESProducer::~BtagPerformanceESProducer() {}

std::unique_ptr<BtagPerformance> BtagPerformanceESProducer::produce(const BTagPerformanceRecord& iRecord) {
  ESHandle<PerformancePayload> pl;
  //ESHandle<PhysicsPerformancePayload> pl;
  ESHandle<PerformanceWorkingPoint> wp;
  iRecord.getRecord<PerformancePayloadRecord>().get(mypl, pl);

  iRecord.getRecord<PerformanceWPRecord>().get(mywp, wp);

  //    BtagWorkingPoint wp;

  return std::make_unique<BtagPerformance>(*((pl.product())), *((wp.product())));
}

DEFINE_FWK_EVENTSETUP_MODULE(BtagPerformanceESProducer);

#include "RecoBTag/PerformanceDB/plugins/BtagPerformanceESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <iostream>
#include <memory>
#include <string>

using namespace edm;

BtagPerformanceESProducer::BtagPerformanceESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  mypl = p.getParameter<std::string>("PayloadName"); 
  mywp = p.getParameter<std::string>("WorkingPointName");
  
  pset_ = p;
  setWhatProduced(this,myname);
}

BtagPerformanceESProducer::~BtagPerformanceESProducer() {}

std::shared_ptr<BtagPerformance> 
BtagPerformanceESProducer::produce(const BTagPerformanceRecord & iRecord){ 
   ESHandle<PerformancePayload> pl;
   //ESHandle<PhysicsPerformancePayload> pl;
   ESHandle<PerformanceWorkingPoint> wp;
   iRecord.getRecord<PerformancePayloadRecord>().get(mypl,pl);
   
   std::cout <<"HERE "<<std::endl;
   iRecord.getRecord<PerformanceWPRecord>().get(mywp,wp);
   std::cout <<"HERE "<<std::endl;
   
   std::cout <<" Got the payload, which is a  "<<typeid(*(pl.product())).name()<<std::endl;
   
   //    BtagWorkingPoint wp;
   
   
   
   _perf  = std::make_shared<BtagPerformance>(*((pl.product())), *((wp.product())));
   //    _perf  = std::make_shared<BtagPerformance>(*((pl.product())), wp);
   return _perf;
}


DEFINE_FWK_EVENTSETUP_MODULE(BtagPerformanceESProducer);


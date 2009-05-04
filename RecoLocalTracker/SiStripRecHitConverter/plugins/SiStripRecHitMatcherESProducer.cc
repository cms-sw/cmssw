#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitMatcherESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"



#include <string>
#include <memory>

using namespace edm;

SiStripRecHitMatcherESProducer::SiStripRecHitMatcherESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

SiStripRecHitMatcherESProducer::~SiStripRecHitMatcherESProducer() {}

boost::shared_ptr<SiStripRecHitMatcher> 
SiStripRecHitMatcherESProducer::produce(const TkStripCPERecord & iRecord){ 

  _matcher  = boost::shared_ptr<SiStripRecHitMatcher>(new SiStripRecHitMatcher(pset_));
  return _matcher;
}



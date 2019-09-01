#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitMatcherESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

SiStripRecHitMatcherESProducer::SiStripRecHitMatcherESProducer(const edm::ParameterSet& p) {
  std::string name = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this, name);
}

std::unique_ptr<SiStripRecHitMatcher> SiStripRecHitMatcherESProducer::produce(const TkStripCPERecord& iRecord) {
  return std::make_unique<SiStripRecHitMatcher>(pset_);
}

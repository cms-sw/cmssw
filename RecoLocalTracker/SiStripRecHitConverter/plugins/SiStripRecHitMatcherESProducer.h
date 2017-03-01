#ifndef RecoLocaltracker_SiStriprecHitConverter_SiStripRecHitMatcherESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_SiStripRecHitMatcherESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include <memory>

class SiStripRecHitMatcherESProducer: public edm::ESProducer {
 public:
  SiStripRecHitMatcherESProducer(const edm::ParameterSet&);
  std::shared_ptr<SiStripRecHitMatcher> produce(const TkStripCPERecord&);
 private:
  std::shared_ptr<SiStripRecHitMatcher> matcher_;
  edm::ParameterSet pset_;
};
#endif





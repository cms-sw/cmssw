#ifndef RecoLocaltracker_SiStriprecHitConverter_SiStripRecHitMatcherESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_SiStripRecHitMatcherESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include <boost/shared_ptr.hpp>

class  SiStripRecHitMatcherESProducer: public edm::ESProducer{
 public:
  SiStripRecHitMatcherESProducer(const edm::ParameterSet & p);
  virtual ~SiStripRecHitMatcherESProducer(); 
  boost::shared_ptr<SiStripRecHitMatcher> produce(const TrackerCPERecord &);
 private:
  boost::shared_ptr<SiStripRecHitMatcher> _matcher;
  edm::ParameterSet pset_;
};


#endif





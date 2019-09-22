#ifndef RecoLocaltracker_SiPhase2VectorHitBuilder_SiPhase2RecHitMatcherESProducer_h
#define RecoLocaltracker_SiPhase2VectorHitBuilder_SiPhase2RecHitMatcherESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderEDProducer.h"
#include <memory>

class SiPhase2RecHitMatcherESProducer: public edm::ESProducer {
 public:
  SiPhase2RecHitMatcherESProducer(const edm::ParameterSet&);
  std::shared_ptr<VectorHitBuilderEDProducer> produce(const TkPhase2OTCPERecord&);
 private:
  std::string name;
  std::shared_ptr<VectorHitBuilderEDProducer> matcher_;
  edm::ParameterSet pset_;
};
#endif





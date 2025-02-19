#ifndef RecoParticleFlow_Configuration_HepMCCopy_H
#define RecoParticleFlow_Configuration_HepMCCopy_H

#include "FWCore/Framework/interface/EDProducer.h"

class ParameterSet;
class Event;
class EventSetup;

class HepMCCopy : public edm::EDProducer
{

 public:

  explicit HepMCCopy(edm::ParameterSet const & p);
  virtual ~HepMCCopy() {}
  virtual void beginRun(edm::Run &, edm::EventSetup const&) {}
  virtual void endRun() {}
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:

};

#endif

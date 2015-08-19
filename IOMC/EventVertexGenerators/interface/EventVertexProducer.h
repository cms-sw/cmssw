#ifndef IOMC_EventVertexGenerators_EventVertexProducer_h
#define IOMC_EventVertexGenerators_EventVertexProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <memory>

class BaseEvtVtxGenerator;

namespace edm {
  class Event;
  class EventSetup;
  class HepMCProduct;
  class LuminosityBlock;
  class ParameterSet;
  class Run;
}

class EventVertexProducer : public edm::EDProducer {
public:
   // ctor & dtor
   explicit EventVertexProducer(edm::ParameterSet const&);
   virtual ~EventVertexProducer();
      
  EventVertexProducer(EventVertexProducer const&) = delete;
  EventVertexProducer& operator=(EventVertexProducer const&) = delete;

private :
  virtual void beginRun(edm::Run const& , edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void produce(edm::Event&, edm::EventSetup const&) override;

  edm::EDGetTokenT<edm::HepMCProduct> sourceToken_;
  std::unique_ptr<BaseEvtVtxGenerator> vertexGenerator_;
};

#endif

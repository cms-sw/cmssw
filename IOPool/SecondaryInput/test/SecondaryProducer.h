#ifndef SecondaryInput_SecondaryProducer_h
#define SecondaryInput_SecondaryProducer_h

/** \class SecondaryProducer
 *
 * \author Bill Tanenbaum
 *
 *
 ************************************************************/

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <memory>

namespace edm {
  class EventPrincipal;
  class ProcessConfiguration;
  class VectorInputSource;

  class SecondaryProducer : public EDProducer {
  public:
    /** standard constructor*/
    explicit SecondaryProducer(ParameterSet const& pset);

    /**Default destructor*/
    virtual ~SecondaryProducer();

    /**Accumulates the pileup events into this event*/
    virtual void produce(Event& e1, EventSetup const& c);

    bool processOneEvent(EventPrincipal const& eventPrincipal, Event& e);

  private:
    virtual void put(Event&) {}
    virtual void beginJob();
    virtual void endJob();
    std::shared_ptr<VectorInputSource> makeSecInput(ParameterSet const& ps);

    std::shared_ptr<ProductRegistry const> productRegistry() const { return get_underlying_safe(productRegistry_); }
    std::shared_ptr<ProductRegistry>& productRegistry() { return get_underlying_safe(productRegistry_); }

    edm::propagate_const<std::shared_ptr<ProductRegistry>> productRegistry_;
    edm::propagate_const<std::shared_ptr<VectorInputSource>> secInput_;
    edm::propagate_const<std::unique_ptr<ProcessConfiguration>> processConfiguration_;
    edm::propagate_const<std::unique_ptr<EventPrincipal>> eventPrincipal_;
    bool sequential_;
    bool specified_;
    bool sameLumiBlock_;
    bool firstEvent_;
    bool firstLoop_;
    EventNumber_t expectedEventNumber_;
  };
}  // namespace edm

#endif

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

#include <memory>

namespace edm {
  class EventPrincipal;
  class ProcessConfiguration;
  class VectorInputSource;

  class SecondaryProducer: public EDProducer {
  public:

    /** standard constructor*/
    explicit SecondaryProducer(ParameterSet const& pset);

    /**Default destructor*/
    virtual ~SecondaryProducer();

    /**Accumulates the pileup events into this event*/
    virtual void produce(Event& e1, EventSetup const& c);

    void processOneEvent(EventPrincipal const& eventPrincipal, Event& e);

  private:

    virtual void put(Event &) {}

    virtual void beginJob();

    virtual void endJob();

    std::shared_ptr<VectorInputSource> makeSecInput(ParameterSet const& ps);

    std::shared_ptr<ProductRegistry> productRegistry_;

    std::shared_ptr<VectorInputSource> const secInput_;

    std::unique_ptr<ProcessConfiguration> processConfiguration_;

    std::unique_ptr<EventPrincipal> eventPrincipal_;

    bool sequential_;

    bool specified_;

    bool sameLumiBlock_;

    bool firstEvent_;

    bool firstLoop_;

    EventNumber_t expectedEventNumber_;
  };
}//edm

#endif

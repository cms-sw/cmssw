#ifndef Framework_EDProducer_h
#define Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.14 2006/06/20 23:13:27 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class EDProducer : public ProducerBase {
  public:
    typedef EDProducer ModuleType;

    EDProducer ();
    void doProduce(Event& e, EventSetup const& c,
		   CurrentProcessingContext const* cpcp);
		   
    virtual ~EDProducer();

    virtual void beginJob(EventSetup const&);
    virtual void endJob();

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('produce').
    CurrentProcessingContext const* currentContext() const;

  private:
    virtual void produce(Event& e, EventSetup const& c) = 0; 

    CurrentProcessingContext const* current_context_;
  };
}

#endif

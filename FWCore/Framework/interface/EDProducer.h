#ifndef Framework_EDProducer_h
#define Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.13 2006/04/20 22:33:21 wmtan Exp $

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
    CurrentProcessingContext const* currentContext() const;

  private:
    virtual void produce(Event& e, EventSetup const& c) = 0; 

    CurrentProcessingContext const* current_context_;
  };
}

#endif

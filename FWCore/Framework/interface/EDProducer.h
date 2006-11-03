#ifndef Framework_EDProducer_h
#define Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.16 2006/10/31 23:54:01 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class EDProducer : public ProducerBase {
  public:
    typedef EDProducer ModuleType;

    EDProducer ();
    virtual ~EDProducer();

    void doProduce(Event& e, EventSetup const& c,
		   CurrentProcessingContext const* cpcp);
    void doBeginJob(EventSetup const&);
    void doEndJob();
    void doBeginRun(Run & r, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doEndRun(Run & r, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doBeginLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doEndLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
		   CurrentProcessingContext const* cpc);


  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('produce').
    CurrentProcessingContext const* currentContext() const;

  private:
    virtual void produce(Event &, EventSetup const&) = 0;
    virtual void beginJob(EventSetup const&){}
    virtual void endJob(){}
    virtual void beginRun(Run &, EventSetup const&){}
    virtual void endRun(Run &, EventSetup const&){}
    virtual void beginLuminosityBlock(LuminosityBlock &, EventSetup const&){}
    virtual void endLuminosityBlock(LuminosityBlock &, EventSetup const&){}

    CurrentProcessingContext const* current_context_;
  };
}

#endif

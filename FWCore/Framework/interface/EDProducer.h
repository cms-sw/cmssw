#ifndef FWCore_Framework_EDProducer_h
#define FWCore_Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.19 2007/09/18 18:06:46 chrjones Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace edm {
  class EDProducer : public ProducerBase {
  public:
    friend class ProducerWorker;
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
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRespondToOpenOutputFiles(FileBlock const& fb);
    void doRespondToCloseOutputFiles(FileBlock const& fb);

    static void fillDescription(edm::ParameterSetDescription&);
    
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
    virtual void respondToOpenInputFile(FileBlock const& fb) {}
    virtual void respondToCloseInputFile(FileBlock const& fb) {}
    virtual void respondToOpenOutputFiles(FileBlock const& fb) {}
    virtual void respondToCloseOutputFiles(FileBlock const& fb) {}

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;
    CurrentProcessingContext const* current_context_;
  };
}

#endif

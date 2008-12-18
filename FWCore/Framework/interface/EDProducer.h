#ifndef FWCore_Framework_EDProducer_h
#define FWCore_Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

namespace edm {
  class EDProducer : public ProducerBase {
  public:
    template <typename T> friend class WorkerT;
    typedef EDProducer ModuleType;
    typedef WorkerT<EDProducer> WorkerType;

    EDProducer ();
    virtual ~EDProducer();

    static void fillDescription(edm::ParameterSetDescription& iDesc,
                                std::string const& moduleLabel);

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('produce').
    CurrentProcessingContext const* currentContext() const;

  private:
    bool doEvent(EventPrincipal& ep, EventSetup const& c,
		   CurrentProcessingContext const* cpcp);
    void doBeginJob(EventSetup const&);
    void doEndJob();
    bool doBeginRun(RunPrincipal & rp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndRun(RunPrincipal & rp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal & lbp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal & lbp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRespondToOpenOutputFiles(FileBlock const& fb);
    void doRespondToCloseOutputFiles(FileBlock const& fb);
    void registerAnyProducts(boost::shared_ptr<EDProducer>& module, ProductRegistry *reg) {
      registerProducts(module, reg, moduleDescription_);
    }

    std::string workerType() const {return "WorkerT<EDProducer>";}

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

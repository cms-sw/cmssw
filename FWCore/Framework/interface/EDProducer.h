#ifndef FWCore_Framework_EDProducer_h
#define FWCore_Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>
#include <vector>

namespace edm {
  class EDProducer : public ProducerBase {
  public:
    template <typename T> friend class WorkerT;
    typedef EDProducer ModuleType;
    typedef WorkerT<EDProducer> WorkerType;

    EDProducer ();
    virtual ~EDProducer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('produce').
    CurrentProcessingContext const* currentContext() const;

  private:
    bool doEvent(EventPrincipal& ep, EventSetup const& c,
		   CurrentProcessingContext const* cpcp);
    void doBeginJob();
    void doEndJob();
    bool doBeginRun(RunPrincipal& rp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndRun(RunPrincipal& rp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRespondToOpenOutputFiles(FileBlock const& fb);
    void doRespondToCloseOutputFiles(FileBlock const& fb);
    void doPreForkReleaseResources();
    void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);
    void registerAnyProducts(EDProducer* module, ProductRegistry* reg) {
      registerProducts(module, reg, moduleDescription_);
    }

    std::string workerType() const {return "WorkerT<EDProducer>";}

    virtual void produce(Event&, EventSetup const&) = 0;
    virtual void beginJob() {}
    virtual void endJob(){}
    virtual void beginRun(Run&, EventSetup const&){}
    virtual void endRun(Run&, EventSetup const&){}
    virtual void beginLuminosityBlock(LuminosityBlock&, EventSetup const&){}
    virtual void endLuminosityBlock(LuminosityBlock&, EventSetup const&){}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}
    virtual void respondToOpenOutputFiles(FileBlock const&) {}
    virtual void respondToCloseOutputFiles(FileBlock const&) {}
    virtual void preForkReleaseResources() {}
    virtual void postForkReacquireResources(unsigned int /*iChildIndex*/, unsigned int /*iNumberOfChildren*/) {}

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;
    CurrentProcessingContext const* current_context_;
    std::vector<BranchID> previousParentage_;
    ParentageID previousParentageId_;
  };
}

#endif

#ifndef FWCore_Framework_EDAnalyzer_h
#define FWCore_Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

// EDAnalyzer is the base class for all analyzer "modules".

namespace edm {

  class EDAnalyzer {
  public:
    template <typename T> friend class WorkerT;
    typedef EDAnalyzer ModuleType;
    typedef WorkerT<EDAnalyzer> WorkerType;

    EDAnalyzer() : moduleDescription_(), current_context_(0) {}
    virtual ~EDAnalyzer();
    
    std::string workerType() const {return "WorkerT<EDAnalyzer>";}

    static void fillDescription(edm::ParameterSetDescription& iDesc,
                                std::string const& moduleLabel);

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('analyze').
    CurrentProcessingContext const* currentContext() const;

  private:
    bool doEvent(EventPrincipal const& ep, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doBeginJob(EventSetup const&);
    void doEndJob();
    bool doBeginRun(RunPrincipal const& rp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndRun(RunPrincipal const& rp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRespondToOpenOutputFiles(FileBlock const& fb);
    void doRespondToCloseOutputFiles(FileBlock const& fb);
    void registerAnyProducts(boost::shared_ptr<EDAnalyzer>const&, ProductRegistry const*) {}

    virtual void analyze(Event const&, EventSetup const&) = 0;
    virtual void beginJob(EventSetup const&){}
    virtual void endJob(){}
    virtual void beginRun(Run const&, EventSetup const&){}
    virtual void endRun(Run const&, EventSetup const&){}
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
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

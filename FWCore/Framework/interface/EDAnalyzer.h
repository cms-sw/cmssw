#ifndef FWCore_Framework_EDAnalyzer_h
#define FWCore_Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
// EDAnalyzer is the base class for all reconstruction "modules".

namespace edm {

  class EDAnalyzer {
  public:
    template <typename T> friend class WorkerT;
    friend class AnalyzerWorker;
    EDAnalyzer() : moduleDescription_(), current_context_(0) {}
    virtual ~EDAnalyzer();

    typedef EDAnalyzer ModuleType;
    
    void doAnalyze(Event const& e, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doBeginJob(EventSetup const&);
    void doEndJob();
    void doBeginRun(Run const& r, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doEndRun(Run const& r, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doBeginLuminosityBlock(LuminosityBlock const& lb, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doEndLuminosityBlock(LuminosityBlock const& lb, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRespondToOpenOutputFiles(FileBlock const& fb);
    void doRespondToCloseOutputFiles(FileBlock const& fb);

    static void fillDescription(edm::ParameterSetDescription&);

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('analyze').
    CurrentProcessingContext const* currentContext() const;

  private:
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

#ifndef Framework_EDAnalyzer_h
#define Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"

// EDAnalyzer is the base class for all reconstruction "modules".

namespace edm {

  class EDAnalyzer 
  {
  public:

    EDAnalyzer() : current_context_(0) {}
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

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('analyze').
    CurrentProcessingContext const* currentContext() const;

  private:
    virtual void analyze(Event const&, EventSetup const&) {}
    virtual void beginJob(EventSetup const&) {}
    virtual void endJob() {}
    virtual void beginRun(Run const&, EventSetup const&) {}
    virtual void endRun(Run const&, EventSetup const&) {}
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}

    CurrentProcessingContext const* current_context_;
  };
}

#endif

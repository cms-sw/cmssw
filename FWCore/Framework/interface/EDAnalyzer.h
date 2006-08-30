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

    void doBeginJob(EventSetup const&) ;
    void doEndJob() ;


  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('analyze').
    CurrentProcessingContext const* currentContext() const;


  private:
    virtual void analyze(Event const& e, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&) ;
    virtual void endJob() ;

    CurrentProcessingContext const* current_context_;
  };
}

#endif

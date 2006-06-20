#ifndef Framework_EDAnalyzer_h
#define Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"

// EDAnalyzer is the base class for all reconstruction "modules".

namespace edm {

  class EDAnalyzer 
  {
  public:
    typedef EDAnalyzer ModuleType;
    
    void doAnalyze(Event const& e, EventSetup const& c,
		   CurrentProcessingContext const* cpc);

    void doBeginJob(EventSetup const&) ;
    void doEndJob() ;
    virtual ~EDAnalyzer();

  private:
    virtual void analyze(Event const& e, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&) ;
    virtual void endJob() ;

    CurrentProcessingContext const* current_context_;
  };
}

#endif

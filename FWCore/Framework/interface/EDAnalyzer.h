#ifndef Framework_EDAnalyzer_h
#define Framework_EDAnalyzer_h

// EDAnalyzer is the base class for all reconstruction "modules".

#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm
  {
  class EDAnalyzer
    {
    public:
      typedef EDAnalyzer ModuleType;

      virtual ~EDAnalyzer();
      virtual void analyze(Event const& e, EventSetup const& c) = 0;
      virtual void beginJob(EventSetup const&) ;
      virtual void endJob() ;
    };
}

#endif // Framework_EDAnalyzer_h

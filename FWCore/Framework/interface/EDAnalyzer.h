#ifndef EDM_EDANALYZER_INCLUDED
#define EDM_EDANALYZER_INCLUDED

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
      virtual void beginJob( EventSetup const& ) ;
      virtual void endJob() ;
    };
}

#endif // EDM_EDANALYZER_INCLUDED

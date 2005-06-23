#ifndef EDM_EDANALYZER_INCLUDED
#define EDM_EDANALYZER_INCLUDED

// EDAnalyzer is the base class for all reconstruction "modules".

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"

namespace edm
  {
  class EDAnalyzer
    {
    public:
      typedef EDAnalyzer ModuleType;

      virtual ~EDAnalyzer();
      virtual void analyze(Event const& e, EventSetup const& c) = 0;
    };
}

#endif // EDM_EDANALYZER_INCLUDED

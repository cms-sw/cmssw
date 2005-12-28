#ifndef Framework_EDAnalyzer_h
#define Framework_EDAnalyzer_h

// EDAnalyzer is the base class for all reconstruction "modules".

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class EDAnalyzer {
    public:
      typedef EDAnalyzer ModuleType;

      virtual ~EDAnalyzer();
      virtual void analyze(Event const& e, EventSetup const& c) = 0;
      virtual void beginJob(EventSetup const&) ;
      virtual void endJob() ;
    };
}

#endif

#ifndef FWCore_Framework_EventSetupProviderMaker_h
#define FWCore_Framework_EventSetupProviderMaker_h

// system include files
#include <memory>

// forward declarations
namespace edm {
  class ParameterSet;
  namespace eventsetup {
    class EventSetupProvider;

    std::auto_ptr<EventSetupProvider>
    makeEventSetupProvider(ParameterSet const& params);

    void
    fillEventSetupProvider(EventSetupProvider& cp,
                           ParameterSet& params);

    void
    validateEventSetupParameters(ParameterSet& pset);
  }
}
#endif

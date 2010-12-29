#ifndef Framework_EventSetupProviderMaker_h
#define Framework_EventSetupProviderMaker_h

// system include files
#include <memory>

// forward declarations
namespace edm {
  class CommonParams;
  class ParameterSet;
  namespace eventsetup {
    class EventSetupProvider;

    std::auto_ptr<EventSetupProvider>
    makeEventSetupProvider(ParameterSet const& params);

    void
    fillEventSetupProvider(EventSetupProvider& cp,
                           ParameterSet& params,
                           CommonParams const& common);
  }
}
#endif

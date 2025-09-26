#ifndef FWCore_Framework_EventSetupProviderMaker_h
#define FWCore_Framework_EventSetupProviderMaker_h

// system include files
#include <memory>

// forward declarations
namespace edm {
  class ActivityRegistry;
  class ModuleTypeResolverMaker;
  class ParameterSet;
  namespace eventsetup {
    class EventSetupProvider;

    std::unique_ptr<EventSetupProvider> makeEventSetupProvider(ParameterSet const& params, ActivityRegistry*);

    void fillEventSetupProvider(ModuleTypeResolverMaker const* resolverMaker,
                                EventSetupProvider& cp,
                                ParameterSet& params);
  }  // namespace eventsetup
}  // namespace edm
#endif

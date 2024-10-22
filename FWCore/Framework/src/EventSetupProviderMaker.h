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
    class EventSetupsController;

    std::unique_ptr<EventSetupProvider> makeEventSetupProvider(ParameterSet const& params,
                                                               unsigned subProcessIndex,
                                                               ActivityRegistry*);

    void fillEventSetupProvider(ModuleTypeResolverMaker const* resolverMaker,
                                EventSetupsController& esController,
                                EventSetupProvider& cp,
                                ParameterSet& params);
  }  // namespace eventsetup
}  // namespace edm
#endif

#ifndef FWCore_Framework_EventSetupProviderMaker_h
#define FWCore_Framework_EventSetupProviderMaker_h

// system include files
#include <memory>
#include <vector>

// forward declarations
namespace edm {
  class ActivityRegistry;
  class ModuleTypeResolverMaker;
  class ParameterSet;
  class EventSetupRecordIntervalFinder;
  namespace eventsetup {
    class EventSetupProvider;

    std::unique_ptr<EventSetupProvider> makeEventSetupProvider(ParameterSet const& params, ActivityRegistry*);

    std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>> fillEventSetupProvider(
        ModuleTypeResolverMaker const* resolverMaker, EventSetupProvider& cp, ParameterSet& params);
  }  // namespace eventsetup
}  // namespace edm
#endif

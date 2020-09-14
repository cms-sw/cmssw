#ifndef FWCore_Framework_EventSetupProviderMaker_h
#define FWCore_Framework_EventSetupProviderMaker_h

// system include files
#include <memory>

// forward declarations
namespace edm {
  class ActivityRegistry;
  class ParameterSet;
  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupsController;

    std::unique_ptr<EventSetupProvider> makeEventSetupProvider(ParameterSet const& params,
                                                               unsigned subProcessIndex,
                                                               ActivityRegistry*);

    void fillEventSetupProvider(EventSetupsController& esController, EventSetupProvider& cp, ParameterSet& params);

    void validateEventSetupParameters(ParameterSet& pset);
  }  // namespace eventsetup
}  // namespace edm
#endif

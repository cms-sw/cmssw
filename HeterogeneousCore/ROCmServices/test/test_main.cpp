#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

class ServiceRegistryListener : public Catch::EventListenerBase {
public:
  using Catch::EventListenerBase::EventListenerBase;  // inherit constructor

  void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
};

CATCH_REGISTER_LISTENER(ServiceRegistryListener);

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

class ServiceRegistryListener : public Catch::TestEventListenerBase {
public:
  using Catch::TestEventListenerBase::TestEventListenerBase;  // inherit constructor

  void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
};

CATCH_REGISTER_LISTENER(ServiceRegistryListener);

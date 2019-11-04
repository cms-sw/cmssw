#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

class ServiceRegistryListener : public Catch::TestEventListenerBase {
public:
  using Catch::TestEventListenerBase::TestEventListenerBase;  // inherit constructor

  void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    const std::string config{
        R"_(import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.CUDAService = cms.Service('CUDAService')
)_"};

    std::unique_ptr<edm::ParameterSet> params;
    edm::makeParameterSets(config, params);
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));
    operate_.reset(new edm::ServiceRegistry::Operate(tempToken));
  }

private:
  std::unique_ptr<edm::ServiceRegistry::Operate> operate_;
};
CATCH_REGISTER_LISTENER(ServiceRegistryListener);

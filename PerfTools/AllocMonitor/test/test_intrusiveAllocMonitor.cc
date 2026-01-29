#include "FWCore/AbstractServices/interface/IntrusiveMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include <cassert>
#include <memory>
#include <vector>

std::unique_ptr<int> nested() {
  edm::Service<edm::IntrusiveMonitorBase> imb;
  auto guard = imb->startMonitoring("Nested allocation inner");

  return std::make_unique<int>(42);
}

int main() {
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string const config = R"_(
import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.add_(cms.Service('IntrusiveAllocMonitor'))
)_";
  std::unique_ptr<edm::ParameterSet> params;
  edm::makeParameterSets(config, params);
  auto token = edm::ServiceToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));
  edm::ServiceRegistry::Operate operate(token);

  edm::Service<edm::IntrusiveMonitorBase> imb;
  assert(imb.isAvailable());

  std::vector<int> vec;
  constexpr int N = 10000;
  {
    auto guard = imb->startMonitoring("Vector fill");
    for (int i = 0; i < N; ++i) {
      vec.push_back(i * 2 - 1);
    }

    int sum = 0;
    for (int a : vec) {
      sum += a;
    }

    edm::LogPrint("Test").format("Sum {}", sum);
  }
  {
    auto guard = imb->startMonitoring(std::string("Vector fill again"));
    for (int i = 0; i < N; ++i) {
      vec.push_back(i * 2 - 1);
    }

    int sum = 0;
    for (int a : vec) {
      sum += a;
    }

    edm::LogPrint("Test").format("Sum {}", sum);
  }

  vec = std::vector<int>();  // clear capacity
  {
    int sum = 0;
    {
      auto guard = imb->startMonitoring("Nested allocation outer vector fill");
      for (int i = 0; i < N; ++i) {
        vec.push_back(i * 2 - 1);
      }

      auto ptr = nested();
      for (int a : vec) {
        sum += a - *ptr;
      }
    }

    edm::LogPrint("Test").format("Sum {}", sum);
  }

  return 0;
}

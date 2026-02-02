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
  auto guard = imb->startMonitoring("inner unique_ptr");

  return std::make_unique<int>(42);
}

// dirty tricks to prevent compiler from optimizing allocations away...
std::atomic<int*> ptr1 = nullptr;
std::atomic<int*> ptr2 = nullptr;
std::atomic<int*> ptr3 = nullptr;
std::atomic<int*> ptr4 = nullptr;

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
  edm::LogPrint("Test").format("Test vector fill");
  {
    int sum = 0;
    {
      auto guard = imb->startMonitoring("Vector fill");
      for (int i = 0; i < N; ++i) {
        vec.push_back(i * 2 - 1);
      }

      for (int a : vec) {
        sum += a;
      }
    }

    edm::LogPrint("Test").format("Sum {}", sum);
  }
  edm::LogPrint("Test").format("====================");

  edm::LogPrint("Test").format("Test vector fill again");
  {
    int sum = 0;
    {
      auto guard = imb->startMonitoring(std::string("Vector fill again"));
      for (int i = 0; i < N; ++i) {
        vec.push_back(i * 2 - 1);
      }

      for (int a : vec) {
        sum += a;
      }
    }

    edm::LogPrint("Test").format("Sum {}", sum);
  }
  edm::LogPrint("Test").format("====================");

  edm::LogPrint("Test").format("Test nested allocation with empty outer");
  {
    int sum = 0;
    {
      auto guard = imb->startMonitoring("Nested allocation empty outer");
      auto ptr2 = nested();
      sum = *ptr2;
    }
    edm::LogPrint("Test").format("Sum {}", sum);
  }
  edm::LogPrint("Test").format("====================");

  edm::LogPrint("Test").format("Test nested allocation");
  {
    int sum = 0;
    {
      auto guard = imb->startMonitoring("Nested allocation outer unique_ptr");
      auto ptr1 = std::make_unique<int>(42);
      auto ptr2 = nested();
      sum = *ptr1 + *ptr2;
    }

    edm::LogPrint("Test").format("Sum {}", sum);
  }
  edm::LogPrint("Test").format("====================");

  edm::LogPrint("Test").format("Test nested with string messages");
  {
    int sum = 0;
    {
      auto guard1 = imb->startMonitoring("Nested allocation with string messages outer unique_ptr");
      ptr1 = new int(42);
      ptr2 = nullptr;
      {
        auto guard2 = imb->startMonitoring(std::string("inner unique_ptr"));
        ptr2 = new int(67);
      }
      ptr3 = nullptr;
      {
        auto guard3 = imb->startMonitoring(std::string("another inner unique_ptr"));
        ptr3 = new int(11);
      }
      ptr4 = nullptr;
      {
        auto guard4 = imb->startMonitoring(std::string("inner empty"));
        {
          auto guard4_1 = imb->startMonitoring(std::string("inner unique_ptr"));
          ptr4 = new int(313);
        }
      }
      sum = *ptr1 + *ptr2 + *ptr3 + *ptr4;
      delete ptr4.load();
      delete ptr3.load();
      delete ptr2.load();
      delete ptr1.load();
    }
    edm::LogPrint("Test").format("Sum {}", sum);
  }
  edm::LogPrint("Test").format("====================");

  return 0;
}

// -*- C++ -*-
//
// Package:     Services
// Class  :     TestService
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  W. David Dagenhart
//         Created:  14 July 2021

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace edm {
  namespace service {

    class TestService {
    public:
      TestService(const ParameterSet&, ActivityRegistry&);

      static void fillDescriptions(edm::ConfigurationDescriptions&);

      void preBeginProcessBlock(GlobalContext const&);

      void preEndProcessBlock(GlobalContext const&);

      void preGlobalBeginRun(GlobalContext const&);

      void preGlobalEndRun(GlobalContext const&);

      void preGlobalBeginLumi(GlobalContext const&);

      void preGlobalEndLumi(GlobalContext const&);

    private:
      bool printTestMessageLoggerErrors_;
    };
  }  // namespace service
}  // namespace edm

using namespace edm::service;

TestService::TestService(ParameterSet const& iPS, ActivityRegistry& iRegistry)
    : printTestMessageLoggerErrors_(iPS.getUntrackedParameter<bool>("printTestMessageLoggerErrors")) {
  iRegistry.watchPreBeginProcessBlock(this, &TestService::preBeginProcessBlock);

  iRegistry.watchPreEndProcessBlock(this, &TestService::preEndProcessBlock);

  iRegistry.watchPreGlobalBeginRun(this, &TestService::preGlobalBeginRun);

  iRegistry.watchPreGlobalEndRun(this, &TestService::preGlobalEndRun);

  iRegistry.watchPreGlobalBeginLumi(this, &TestService::preGlobalBeginLumi);

  iRegistry.watchPreGlobalEndLumi(this, &TestService::preGlobalEndLumi);
}

void TestService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("printTestMessageLoggerErrors", false)
      ->setComment("Prints MessageLogger errors to test formatting of such messages when printed from Services");
  descriptions.add("TestService", desc);
}

void TestService::preBeginProcessBlock(GlobalContext const&) {
  if (printTestMessageLoggerErrors_) {
    edm::LogError("TestMessageLogger") << "test message from TestService::preBeginProcessBlock";
  }
}

void TestService::preEndProcessBlock(GlobalContext const&) {
  if (printTestMessageLoggerErrors_) {
    edm::LogError("TestMessageLogger") << "test message from TestService::preEndProcessBlock";
  }
}

void TestService::preGlobalBeginRun(GlobalContext const&) {
  if (printTestMessageLoggerErrors_) {
    edm::LogError("TestMessageLogger") << "test message from TestService::preGlobalBeginRun";
  }
}

void TestService::preGlobalEndRun(GlobalContext const&) {
  if (printTestMessageLoggerErrors_) {
    edm::LogError("TestMessageLogger") << "test message from TestService::preGlobalEndRun";
  }
}

void TestService::preGlobalBeginLumi(GlobalContext const& gc) {
  if (printTestMessageLoggerErrors_) {
    edm::LogError("TestMessageLogger") << "test message from TestService::preGlobalBeginLumi";
  }
}

void TestService::preGlobalEndLumi(GlobalContext const& gc) {
  if (printTestMessageLoggerErrors_) {
    edm::LogError("TestMessageLogger") << "test message from TestService::preGlobalEndLumi";
  }
}

using edm::service::TestService;
DEFINE_FWK_SERVICE(TestService);

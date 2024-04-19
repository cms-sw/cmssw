// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     TestServiceOne
//
// Implementation:
//     Service initially intended for testing behavior after exceptions.
//     ExceptionThrowingProducer uses this and is in the same test plugin
//     library and could be used to access the service if it was ever useful
//     for debugging issues related to begin/end transitions.
//
// Original Author:  W. David Dagenhart
//         Created:  13 March 2024

#include "FWCore/Integration/plugins/TestServiceOne.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace edmtest {

  TestServiceOne::TestServiceOne(edm::ParameterSet const& iPS, edm::ActivityRegistry& iRegistry)
      : verbose_(iPS.getUntrackedParameter<bool>("verbose")) {
    iRegistry.watchPreBeginProcessBlock(this, &TestServiceOne::preBeginProcessBlock);
    iRegistry.watchPreEndProcessBlock(this, &TestServiceOne::preEndProcessBlock);

    iRegistry.watchPreGlobalBeginRun(this, &TestServiceOne::preGlobalBeginRun);
    iRegistry.watchPreGlobalEndRun(this, &TestServiceOne::preGlobalEndRun);
    iRegistry.watchPreGlobalBeginLumi(this, &TestServiceOne::preGlobalBeginLumi);
    iRegistry.watchPreGlobalEndLumi(this, &TestServiceOne::preGlobalEndLumi);

    iRegistry.watchPreStreamBeginLumi(this, &TestServiceOne::preStreamBeginLumi);
    iRegistry.watchPostStreamBeginLumi(this, &TestServiceOne::postStreamBeginLumi);
    iRegistry.watchPreStreamEndLumi(this, &TestServiceOne::preStreamEndLumi);
    iRegistry.watchPostStreamEndLumi(this, &TestServiceOne::postStreamEndLumi);

    iRegistry.watchPreModuleStreamBeginLumi(this, &TestServiceOne::preModuleStreamBeginLumi);
    iRegistry.watchPostModuleStreamBeginLumi(this, &TestServiceOne::postModuleStreamBeginLumi);
    iRegistry.watchPreModuleStreamEndLumi(this, &TestServiceOne::preModuleStreamEndLumi);
    iRegistry.watchPostModuleStreamEndLumi(this, &TestServiceOne::postModuleStreamEndLumi);
  }

  void TestServiceOne::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false)->setComment("Prints LogAbsolute messages if true");
    descriptions.add("TestServiceOne", desc);
  }

  void TestServiceOne::preBeginProcessBlock(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preBeginProcessBlock";
    }
  }

  void TestServiceOne::preEndProcessBlock(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preEndProcessBlock";
    }
  }

  void TestServiceOne::preGlobalBeginRun(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preGlobalBeginRun";
    }
  }

  void TestServiceOne::preGlobalEndRun(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preGlobalEndRun";
    }
  }

  void TestServiceOne::preGlobalBeginLumi(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preGlobalBeginLumi";
    }
  }

  void TestServiceOne::preGlobalEndLumi(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preGlobalEndLumi";
    }
  }

  void TestServiceOne::preStreamBeginLumi(edm::StreamContext const&) {
    ++nPreStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preStreamBeginLumi";
    }
  }

  void TestServiceOne::postStreamBeginLumi(edm::StreamContext const&) {
    ++nPostStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::postStreamBeginLumi";
    }
  }

  void TestServiceOne::preStreamEndLumi(edm::StreamContext const&) {
    ++nPreStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preStreamEndLumi";
    }
  }

  void TestServiceOne::postStreamEndLumi(edm::StreamContext const&) {
    ++nPostStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::postStreamEndLumi";
    }
  }

  void TestServiceOne::preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPreModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preModuleStreamBeginLumi";
    }
  }

  void TestServiceOne::postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPostModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::postModuleStreamBeginLumi";
    }
  }

  void TestServiceOne::preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPreModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preModuleStreamEndLumi";
    }
  }

  void TestServiceOne::postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPostModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::postModuleStreamEndLumi";
    }
  }

  unsigned int TestServiceOne::nPreStreamBeginLumi() const { return nPreStreamBeginLumi_.load(); }
  unsigned int TestServiceOne::nPostStreamBeginLumi() const { return nPostStreamBeginLumi_.load(); }
  unsigned int TestServiceOne::nPreStreamEndLumi() const { return nPreStreamEndLumi_.load(); }
  unsigned int TestServiceOne::nPostStreamEndLumi() const { return nPostStreamEndLumi_.load(); }

  unsigned int TestServiceOne::nPreModuleStreamBeginLumi() const { return nPreModuleStreamBeginLumi_.load(); }
  unsigned int TestServiceOne::nPostModuleStreamBeginLumi() const { return nPostModuleStreamBeginLumi_.load(); }
  unsigned int TestServiceOne::nPreModuleStreamEndLumi() const { return nPreModuleStreamEndLumi_.load(); }
  unsigned int TestServiceOne::nPostModuleStreamEndLumi() const { return nPostModuleStreamEndLumi_.load(); }
}  // namespace edmtest

using edmtest::TestServiceOne;
DEFINE_FWK_SERVICE(TestServiceOne);

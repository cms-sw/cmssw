// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     TestServiceTwo
//
// Implementation:
//     Service initially intended for testing behavior after exceptions.
//     ExceptionThrowingProducer uses this and is in the same test plugin
//     library and could be used to access the service if it was ever useful
//     for debugging issues related to begin/end transitions.
//
// Original Author:  W. David Dagenhart
//         Created:  13 March 2024

#include "FWCore/Integration/plugins/TestServiceTwo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace edmtest {

  TestServiceTwo::TestServiceTwo(edm::ParameterSet const& iPS, edm::ActivityRegistry& iRegistry)
      : verbose_(iPS.getUntrackedParameter<bool>("verbose")) {
    iRegistry.watchPreBeginProcessBlock(this, &TestServiceTwo::preBeginProcessBlock);
    iRegistry.watchPreEndProcessBlock(this, &TestServiceTwo::preEndProcessBlock);

    iRegistry.watchPreGlobalBeginRun(this, &TestServiceTwo::preGlobalBeginRun);
    iRegistry.watchPreGlobalEndRun(this, &TestServiceTwo::preGlobalEndRun);
    iRegistry.watchPreGlobalBeginLumi(this, &TestServiceTwo::preGlobalBeginLumi);
    iRegistry.watchPreGlobalEndLumi(this, &TestServiceTwo::preGlobalEndLumi);

    iRegistry.watchPreStreamBeginLumi(this, &TestServiceTwo::preStreamBeginLumi);
    iRegistry.watchPostStreamBeginLumi(this, &TestServiceTwo::postStreamBeginLumi);
    iRegistry.watchPreStreamEndLumi(this, &TestServiceTwo::preStreamEndLumi);
    iRegistry.watchPostStreamEndLumi(this, &TestServiceTwo::postStreamEndLumi);

    iRegistry.watchPreModuleStreamBeginLumi(this, &TestServiceTwo::preModuleStreamBeginLumi);
    iRegistry.watchPostModuleStreamBeginLumi(this, &TestServiceTwo::postModuleStreamBeginLumi);
    iRegistry.watchPreModuleStreamEndLumi(this, &TestServiceTwo::preModuleStreamEndLumi);
    iRegistry.watchPostModuleStreamEndLumi(this, &TestServiceTwo::postModuleStreamEndLumi);
  }

  void TestServiceTwo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false)->setComment("Prints LogAbsolute messages if true");
    descriptions.add("TestServiceTwo", desc);
  }

  void TestServiceTwo::preBeginProcessBlock(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preBeginProcessBlock";
    }
  }

  void TestServiceTwo::preEndProcessBlock(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preEndProcessBlock";
    }
  }

  void TestServiceTwo::preGlobalBeginRun(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preGlobalBeginRun";
    }
  }

  void TestServiceTwo::preGlobalEndRun(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preGlobalEndRun";
    }
  }

  void TestServiceTwo::preGlobalBeginLumi(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preGlobalBeginLumi";
    }
  }

  void TestServiceTwo::preGlobalEndLumi(edm::GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preGlobalEndLumi";
    }
  }

  void TestServiceTwo::preStreamBeginLumi(edm::StreamContext const&) {
    ++nPreStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preStreamBeginLumi";
    }
  }

  void TestServiceTwo::postStreamBeginLumi(edm::StreamContext const&) {
    ++nPostStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::postStreamBeginLumi";
    }
  }

  void TestServiceTwo::preStreamEndLumi(edm::StreamContext const&) {
    ++nPreStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preStreamEndLumi";
    }
  }

  void TestServiceTwo::postStreamEndLumi(edm::StreamContext const&) {
    ++nPostStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::postStreamEndLumi";
    }
  }

  void TestServiceTwo::preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPreModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preModuleStreamBeginLumi";
    }
  }

  void TestServiceTwo::postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPostModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::postModuleStreamBeginLumi";
    }
  }

  void TestServiceTwo::preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPreModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preModuleStreamEndLumi";
    }
  }

  void TestServiceTwo::postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&) {
    ++nPostModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::postModuleStreamEndLumi";
    }
  }

  unsigned int TestServiceTwo::nPreStreamBeginLumi() const { return nPreStreamBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPostStreamBeginLumi() const { return nPostStreamBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPreStreamEndLumi() const { return nPreStreamEndLumi_.load(); }
  unsigned int TestServiceTwo::nPostStreamEndLumi() const { return nPostStreamEndLumi_.load(); }

  unsigned int TestServiceTwo::nPreModuleStreamBeginLumi() const { return nPreModuleStreamBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPostModuleStreamBeginLumi() const { return nPostModuleStreamBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPreModuleStreamEndLumi() const { return nPreModuleStreamEndLumi_.load(); }
  unsigned int TestServiceTwo::nPostModuleStreamEndLumi() const { return nPostModuleStreamEndLumi_.load(); }
}  // namespace edmtest

using edmtest::TestServiceTwo;
DEFINE_FWK_SERVICE(TestServiceTwo);

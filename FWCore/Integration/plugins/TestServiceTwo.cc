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

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <ostream>

using edm::GlobalContext;
using edm::ModuleCallingContext;
using edm::StreamContext;

namespace {

  class TimeStamper {
  public:
    TimeStamper(bool enable) : enabled_(enable) {}

    friend std::ostream& operator<<(std::ostream& out, TimeStamper const& timestamp) {
      if (timestamp.enabled_)
        out << std::setprecision(2) << edm::TimeOfDay() << "  ";
      return out;
    }

  private:
    bool enabled_;
  };

  const char* globalIndent = "    ";
  const char* globalModuleIndent = "        ";
  const char* streamIndent = "            ";
  const char* streamModuleIndent = "                ";
}  // namespace

namespace edmtest {

  TestServiceTwo::TestServiceTwo(edm::ParameterSet const& iPS, edm::ActivityRegistry& iRegistry)
      : verbose_(iPS.getUntrackedParameter<bool>("verbose")),
        printTimestamps_(iPS.getUntrackedParameter<bool>("printTimestamps")) {
    iRegistry.watchPreBeginProcessBlock(this, &TestServiceTwo::preBeginProcessBlock);
    iRegistry.watchPreEndProcessBlock(this, &TestServiceTwo::preEndProcessBlock);

    iRegistry.watchPreStreamBeginLumi(this, &TestServiceTwo::preStreamBeginLumi);
    iRegistry.watchPostStreamBeginLumi(this, &TestServiceTwo::postStreamBeginLumi);
    iRegistry.watchPreStreamEndLumi(this, &TestServiceTwo::preStreamEndLumi);
    iRegistry.watchPostStreamEndLumi(this, &TestServiceTwo::postStreamEndLumi);

    iRegistry.watchPreModuleStreamBeginLumi(this, &TestServiceTwo::preModuleStreamBeginLumi);
    iRegistry.watchPostModuleStreamBeginLumi(this, &TestServiceTwo::postModuleStreamBeginLumi);
    iRegistry.watchPreModuleStreamEndLumi(this, &TestServiceTwo::preModuleStreamEndLumi);
    iRegistry.watchPostModuleStreamEndLumi(this, &TestServiceTwo::postModuleStreamEndLumi);

    iRegistry.watchPreGlobalBeginLumi(this, &TestServiceTwo::preGlobalBeginLumi);
    iRegistry.watchPostGlobalBeginLumi(this, &TestServiceTwo::postGlobalBeginLumi);
    iRegistry.watchPreGlobalEndLumi(this, &TestServiceTwo::preGlobalEndLumi);
    iRegistry.watchPostGlobalEndLumi(this, &TestServiceTwo::postGlobalEndLumi);

    iRegistry.watchPreModuleGlobalBeginLumi(this, &TestServiceTwo::preModuleGlobalBeginLumi);
    iRegistry.watchPostModuleGlobalBeginLumi(this, &TestServiceTwo::postModuleGlobalBeginLumi);
    iRegistry.watchPreModuleGlobalEndLumi(this, &TestServiceTwo::preModuleGlobalEndLumi);
    iRegistry.watchPostModuleGlobalEndLumi(this, &TestServiceTwo::postModuleGlobalEndLumi);

    iRegistry.watchPreGlobalWriteLumi(this, &TestServiceTwo::preGlobalWriteLumi);
    iRegistry.watchPostGlobalWriteLumi(this, &TestServiceTwo::postGlobalWriteLumi);

    iRegistry.watchPreStreamBeginRun(this, &TestServiceTwo::preStreamBeginRun);
    iRegistry.watchPostStreamBeginRun(this, &TestServiceTwo::postStreamBeginRun);
    iRegistry.watchPreStreamEndRun(this, &TestServiceTwo::preStreamEndRun);
    iRegistry.watchPostStreamEndRun(this, &TestServiceTwo::postStreamEndRun);

    iRegistry.watchPreModuleStreamBeginRun(this, &TestServiceTwo::preModuleStreamBeginRun);
    iRegistry.watchPostModuleStreamBeginRun(this, &TestServiceTwo::postModuleStreamBeginRun);
    iRegistry.watchPreModuleStreamEndRun(this, &TestServiceTwo::preModuleStreamEndRun);
    iRegistry.watchPostModuleStreamEndRun(this, &TestServiceTwo::postModuleStreamEndRun);

    iRegistry.watchPreGlobalBeginRun(this, &TestServiceTwo::preGlobalBeginRun);
    iRegistry.watchPostGlobalBeginRun(this, &TestServiceTwo::postGlobalBeginRun);
    iRegistry.watchPreGlobalEndRun(this, &TestServiceTwo::preGlobalEndRun);
    iRegistry.watchPostGlobalEndRun(this, &TestServiceTwo::postGlobalEndRun);

    iRegistry.watchPreModuleGlobalBeginRun(this, &TestServiceTwo::preModuleGlobalBeginRun);
    iRegistry.watchPostModuleGlobalBeginRun(this, &TestServiceTwo::postModuleGlobalBeginRun);
    iRegistry.watchPreModuleGlobalEndRun(this, &TestServiceTwo::preModuleGlobalEndRun);
    iRegistry.watchPostModuleGlobalEndRun(this, &TestServiceTwo::postModuleGlobalEndRun);

    iRegistry.watchPreGlobalWriteRun(this, &TestServiceTwo::preGlobalWriteRun);
    iRegistry.watchPostGlobalWriteRun(this, &TestServiceTwo::postGlobalWriteRun);
  }

  void TestServiceTwo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false)->setComment("Prints LogAbsolute messages for every transition");
    desc.addUntracked<bool>("printTimestamps", false)
        ->setComment("Include a time stamp in message printed for every transition");
    descriptions.add("TestServiceTwo", desc);
  }

  void TestServiceTwo::preBeginProcessBlock(GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preBeginProcessBlock";
    }
  }

  void TestServiceTwo::preEndProcessBlock(GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceTwo") << "test message from TestServiceTwo::preEndProcessBlock";
    }
  }

  void TestServiceTwo::preStreamBeginLumi(StreamContext const& sc) {
    ++nPreStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::preStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::postStreamBeginLumi(StreamContext const& sc) {
    ++nPostStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::postStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::preStreamEndLumi(StreamContext const& sc) {
    ++nPreStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::preStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::postStreamEndLumi(StreamContext const& sc) {
    ++nPostStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::postStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::preModuleStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::postModuleStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::preModuleStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::postModuleStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceTwo::preGlobalBeginLumi(GlobalContext const& gc) {
    ++nPreGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::preGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::postGlobalBeginLumi(GlobalContext const& gc) {
    ++nPostGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::postGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::preGlobalEndLumi(GlobalContext const& gc) {
    ++nPreGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::preGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::postGlobalEndLumi(GlobalContext const& gc) {
    ++nPostGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::postGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::preModuleGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::postModuleGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::preModuleGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::postModuleGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::preGlobalWriteLumi(GlobalContext const& gc) {
    ++nPreGlobalWriteLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::preGlobalWriteLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::postGlobalWriteLumi(GlobalContext const& gc) {
    ++nPostGlobalWriteLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::postGlobalWriteLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceTwo::preStreamBeginRun(StreamContext const& sc) {
    ++nPreStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::preStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::postStreamBeginRun(StreamContext const& sc) {
    ++nPostStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::postStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::preStreamEndRun(StreamContext const& sc) {
    ++nPreStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::preStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::postStreamEndRun(StreamContext const& sc) {
    ++nPostStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamIndent << "TestServiceTwo::postStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::preModuleStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::postModuleStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::preModuleStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << streamModuleIndent << "TestServiceTwo::postModuleStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceTwo::preGlobalBeginRun(GlobalContext const& gc) {
    ++nPreGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::preGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::postGlobalBeginRun(GlobalContext const& gc) {
    ++nPostGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::postGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::preGlobalEndRun(GlobalContext const& gc) {
    ++nPreGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::preGlobalEndRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::postGlobalEndRun(GlobalContext const& gc) {
    ++nPostGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::postGlobalEndRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::preModuleGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::postModuleGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::preModuleGlobalEndRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalModuleIndent << "TestServiceTwo::postModuleGlobalEndRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::preGlobalWriteRun(GlobalContext const& gc) {
    ++nPreGlobalWriteRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::preGlobalWriteRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceTwo::postGlobalWriteRun(GlobalContext const& gc) {
    ++nPostGlobalWriteRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceTwo");
      out << globalIndent << "TestServiceTwo::postGlobalWriteRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
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

  unsigned int TestServiceTwo::nPreGlobalBeginLumi() const { return nPreGlobalBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPostGlobalBeginLumi() const { return nPostGlobalBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPreGlobalEndLumi() const { return nPreGlobalEndLumi_.load(); }
  unsigned int TestServiceTwo::nPostGlobalEndLumi() const { return nPostGlobalEndLumi_.load(); }

  unsigned int TestServiceTwo::nPreModuleGlobalBeginLumi() const { return nPreModuleGlobalBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPostModuleGlobalBeginLumi() const { return nPostModuleGlobalBeginLumi_.load(); }
  unsigned int TestServiceTwo::nPreModuleGlobalEndLumi() const { return nPreModuleGlobalEndLumi_.load(); }
  unsigned int TestServiceTwo::nPostModuleGlobalEndLumi() const { return nPostModuleGlobalEndLumi_.load(); }

  unsigned int TestServiceTwo::nPreGlobalWriteLumi() const { return nPreGlobalWriteLumi_.load(); }
  unsigned int TestServiceTwo::nPostGlobalWriteLumi() const { return nPostGlobalWriteLumi_.load(); }

  unsigned int TestServiceTwo::nPreStreamBeginRun() const { return nPreStreamBeginRun_.load(); }
  unsigned int TestServiceTwo::nPostStreamBeginRun() const { return nPostStreamBeginRun_.load(); }
  unsigned int TestServiceTwo::nPreStreamEndRun() const { return nPreStreamEndRun_.load(); }
  unsigned int TestServiceTwo::nPostStreamEndRun() const { return nPostStreamEndRun_.load(); }

  unsigned int TestServiceTwo::nPreModuleStreamBeginRun() const { return nPreModuleStreamBeginRun_.load(); }
  unsigned int TestServiceTwo::nPostModuleStreamBeginRun() const { return nPostModuleStreamBeginRun_.load(); }
  unsigned int TestServiceTwo::nPreModuleStreamEndRun() const { return nPreModuleStreamEndRun_.load(); }
  unsigned int TestServiceTwo::nPostModuleStreamEndRun() const { return nPostModuleStreamEndRun_.load(); }

  unsigned int TestServiceTwo::nPreGlobalBeginRun() const { return nPreGlobalBeginRun_.load(); }
  unsigned int TestServiceTwo::nPostGlobalBeginRun() const { return nPostGlobalBeginRun_.load(); }
  unsigned int TestServiceTwo::nPreGlobalEndRun() const { return nPreGlobalEndRun_.load(); }
  unsigned int TestServiceTwo::nPostGlobalEndRun() const { return nPostGlobalEndRun_.load(); }

  unsigned int TestServiceTwo::nPreModuleGlobalBeginRun() const { return nPreModuleGlobalBeginRun_.load(); }
  unsigned int TestServiceTwo::nPostModuleGlobalBeginRun() const { return nPostModuleGlobalBeginRun_.load(); }
  unsigned int TestServiceTwo::nPreModuleGlobalEndRun() const { return nPreModuleGlobalEndRun_.load(); }
  unsigned int TestServiceTwo::nPostModuleGlobalEndRun() const { return nPostModuleGlobalEndRun_.load(); }

  unsigned int TestServiceTwo::nPreGlobalWriteRun() const { return nPreGlobalWriteRun_.load(); }
  unsigned int TestServiceTwo::nPostGlobalWriteRun() const { return nPostGlobalWriteRun_.load(); }
}  // namespace edmtest

using edmtest::TestServiceTwo;
DEFINE_FWK_SERVICE(TestServiceTwo);

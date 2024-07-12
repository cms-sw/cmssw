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

  TestServiceOne::TestServiceOne(edm::ParameterSet const& iPS, edm::ActivityRegistry& iRegistry)
      : verbose_(iPS.getUntrackedParameter<bool>("verbose")),
        printTimestamps_(iPS.getUntrackedParameter<bool>("printTimestamps")) {
    iRegistry.watchPreBeginProcessBlock(this, &TestServiceOne::preBeginProcessBlock);
    iRegistry.watchPreEndProcessBlock(this, &TestServiceOne::preEndProcessBlock);

    iRegistry.watchPreStreamBeginLumi(this, &TestServiceOne::preStreamBeginLumi);
    iRegistry.watchPostStreamBeginLumi(this, &TestServiceOne::postStreamBeginLumi);
    iRegistry.watchPreStreamEndLumi(this, &TestServiceOne::preStreamEndLumi);
    iRegistry.watchPostStreamEndLumi(this, &TestServiceOne::postStreamEndLumi);

    iRegistry.watchPreModuleStreamBeginLumi(this, &TestServiceOne::preModuleStreamBeginLumi);
    iRegistry.watchPostModuleStreamBeginLumi(this, &TestServiceOne::postModuleStreamBeginLumi);
    iRegistry.watchPreModuleStreamEndLumi(this, &TestServiceOne::preModuleStreamEndLumi);
    iRegistry.watchPostModuleStreamEndLumi(this, &TestServiceOne::postModuleStreamEndLumi);

    iRegistry.watchPreGlobalBeginLumi(this, &TestServiceOne::preGlobalBeginLumi);
    iRegistry.watchPostGlobalBeginLumi(this, &TestServiceOne::postGlobalBeginLumi);
    iRegistry.watchPreGlobalEndLumi(this, &TestServiceOne::preGlobalEndLumi);
    iRegistry.watchPostGlobalEndLumi(this, &TestServiceOne::postGlobalEndLumi);

    iRegistry.watchPreModuleGlobalBeginLumi(this, &TestServiceOne::preModuleGlobalBeginLumi);
    iRegistry.watchPostModuleGlobalBeginLumi(this, &TestServiceOne::postModuleGlobalBeginLumi);
    iRegistry.watchPreModuleGlobalEndLumi(this, &TestServiceOne::preModuleGlobalEndLumi);
    iRegistry.watchPostModuleGlobalEndLumi(this, &TestServiceOne::postModuleGlobalEndLumi);

    iRegistry.watchPreGlobalWriteLumi(this, &TestServiceOne::preGlobalWriteLumi);
    iRegistry.watchPostGlobalWriteLumi(this, &TestServiceOne::postGlobalWriteLumi);

    iRegistry.watchPreStreamBeginRun(this, &TestServiceOne::preStreamBeginRun);
    iRegistry.watchPostStreamBeginRun(this, &TestServiceOne::postStreamBeginRun);
    iRegistry.watchPreStreamEndRun(this, &TestServiceOne::preStreamEndRun);
    iRegistry.watchPostStreamEndRun(this, &TestServiceOne::postStreamEndRun);

    iRegistry.watchPreModuleStreamBeginRun(this, &TestServiceOne::preModuleStreamBeginRun);
    iRegistry.watchPostModuleStreamBeginRun(this, &TestServiceOne::postModuleStreamBeginRun);
    iRegistry.watchPreModuleStreamEndRun(this, &TestServiceOne::preModuleStreamEndRun);
    iRegistry.watchPostModuleStreamEndRun(this, &TestServiceOne::postModuleStreamEndRun);

    iRegistry.watchPreGlobalBeginRun(this, &TestServiceOne::preGlobalBeginRun);
    iRegistry.watchPostGlobalBeginRun(this, &TestServiceOne::postGlobalBeginRun);
    iRegistry.watchPreGlobalEndRun(this, &TestServiceOne::preGlobalEndRun);
    iRegistry.watchPostGlobalEndRun(this, &TestServiceOne::postGlobalEndRun);

    iRegistry.watchPreModuleGlobalBeginRun(this, &TestServiceOne::preModuleGlobalBeginRun);
    iRegistry.watchPostModuleGlobalBeginRun(this, &TestServiceOne::postModuleGlobalBeginRun);
    iRegistry.watchPreModuleGlobalEndRun(this, &TestServiceOne::preModuleGlobalEndRun);
    iRegistry.watchPostModuleGlobalEndRun(this, &TestServiceOne::postModuleGlobalEndRun);

    iRegistry.watchPreGlobalWriteRun(this, &TestServiceOne::preGlobalWriteRun);
    iRegistry.watchPostGlobalWriteRun(this, &TestServiceOne::postGlobalWriteRun);
  }

  void TestServiceOne::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false)->setComment("Prints LogAbsolute messages for every transition");
    desc.addUntracked<bool>("printTimestamps", false)
        ->setComment("Include a time stamp in message printed for every transition");
    descriptions.add("TestServiceOne", desc);
  }

  void TestServiceOne::preBeginProcessBlock(GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preBeginProcessBlock";
    }
  }

  void TestServiceOne::preEndProcessBlock(GlobalContext const&) {
    if (verbose_) {
      edm::LogAbsolute("TestServiceOne") << "test message from TestServiceOne::preEndProcessBlock";
    }
  }

  void TestServiceOne::preStreamBeginLumi(StreamContext const& sc) {
    ++nPreStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::preStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::postStreamBeginLumi(StreamContext const& sc) {
    ++nPostStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::postStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::preStreamEndLumi(StreamContext const& sc) {
    ++nPreStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::preStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::postStreamEndLumi(StreamContext const& sc) {
    ++nPostStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::postStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run()
          << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::preModuleStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::postModuleStreamBeginLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::preModuleStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::postModuleStreamEndLumi " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock();
    }
  }

  void TestServiceOne::preGlobalBeginLumi(GlobalContext const& gc) {
    ++nPreGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::preGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::postGlobalBeginLumi(GlobalContext const& gc) {
    ++nPostGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::postGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::preGlobalEndLumi(GlobalContext const& gc) {
    ++nPreGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::preGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::postGlobalEndLumi(GlobalContext const& gc) {
    ++nPostGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::postGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::preModuleGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalBeginLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::postModuleGlobalBeginLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::preModuleGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalEndLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::postModuleGlobalEndLumi " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run()
          << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::preGlobalWriteLumi(GlobalContext const& gc) {
    ++nPreGlobalWriteLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::preGlobalWriteLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::postGlobalWriteLumi(GlobalContext const& gc) {
    ++nPostGlobalWriteLumi_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::postGlobalWriteLumi " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run() << " lumi = " << gc.luminosityBlockID().luminosityBlock();
    }
  }

  void TestServiceOne::preStreamBeginRun(StreamContext const& sc) {
    ++nPreStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::preStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::postStreamBeginRun(StreamContext const& sc) {
    ++nPostStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::postStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::preStreamEndRun(StreamContext const& sc) {
    ++nPreStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::preStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::postStreamEndRun(StreamContext const& sc) {
    ++nPostStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamIndent << "TestServiceOne::postStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::preModuleStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::postModuleStreamBeginRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPreModuleStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::preModuleStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
    ++nPostModuleStreamEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << streamModuleIndent << "TestServiceOne::postModuleStreamEndRun " << TimeStamper(printTimestamps_)
          << " stream = " << sc.streamID() << " label = " << mcc.moduleDescription()->moduleLabel()
          << " run = " << sc.eventID().run();
    }
  }

  void TestServiceOne::preGlobalBeginRun(GlobalContext const& gc) {
    ++nPreGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::preGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::postGlobalBeginRun(GlobalContext const& gc) {
    ++nPostGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::postGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::preGlobalEndRun(GlobalContext const& gc) {
    ++nPreGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::preGlobalEndRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::postGlobalEndRun(GlobalContext const& gc) {
    ++nPostGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::postGlobalEndRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::preModuleGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalBeginRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::postModuleGlobalBeginRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPreModuleGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::preModuleGlobalEndRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
    ++nPostModuleGlobalEndRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalModuleIndent << "TestServiceOne::postModuleGlobalEndRun " << TimeStamper(printTimestamps_)
          << " label = " << mcc.moduleDescription()->moduleLabel() << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::preGlobalWriteRun(GlobalContext const& gc) {
    ++nPreGlobalWriteRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::preGlobalWriteRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
    }
  }

  void TestServiceOne::postGlobalWriteRun(GlobalContext const& gc) {
    ++nPostGlobalWriteRun_;
    if (verbose_) {
      edm::LogAbsolute out("TestServiceOne");
      out << globalIndent << "TestServiceOne::postGlobalWriteRun " << TimeStamper(printTimestamps_)
          << " run = " << gc.luminosityBlockID().run();
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

  unsigned int TestServiceOne::nPreGlobalBeginLumi() const { return nPreGlobalBeginLumi_.load(); }
  unsigned int TestServiceOne::nPostGlobalBeginLumi() const { return nPostGlobalBeginLumi_.load(); }
  unsigned int TestServiceOne::nPreGlobalEndLumi() const { return nPreGlobalEndLumi_.load(); }
  unsigned int TestServiceOne::nPostGlobalEndLumi() const { return nPostGlobalEndLumi_.load(); }

  unsigned int TestServiceOne::nPreModuleGlobalBeginLumi() const { return nPreModuleGlobalBeginLumi_.load(); }
  unsigned int TestServiceOne::nPostModuleGlobalBeginLumi() const { return nPostModuleGlobalBeginLumi_.load(); }
  unsigned int TestServiceOne::nPreModuleGlobalEndLumi() const { return nPreModuleGlobalEndLumi_.load(); }
  unsigned int TestServiceOne::nPostModuleGlobalEndLumi() const { return nPostModuleGlobalEndLumi_.load(); }

  unsigned int TestServiceOne::nPreGlobalWriteLumi() const { return nPreGlobalWriteLumi_.load(); }
  unsigned int TestServiceOne::nPostGlobalWriteLumi() const { return nPostGlobalWriteLumi_.load(); }

  unsigned int TestServiceOne::nPreStreamBeginRun() const { return nPreStreamBeginRun_.load(); }
  unsigned int TestServiceOne::nPostStreamBeginRun() const { return nPostStreamBeginRun_.load(); }
  unsigned int TestServiceOne::nPreStreamEndRun() const { return nPreStreamEndRun_.load(); }
  unsigned int TestServiceOne::nPostStreamEndRun() const { return nPostStreamEndRun_.load(); }

  unsigned int TestServiceOne::nPreModuleStreamBeginRun() const { return nPreModuleStreamBeginRun_.load(); }
  unsigned int TestServiceOne::nPostModuleStreamBeginRun() const { return nPostModuleStreamBeginRun_.load(); }
  unsigned int TestServiceOne::nPreModuleStreamEndRun() const { return nPreModuleStreamEndRun_.load(); }
  unsigned int TestServiceOne::nPostModuleStreamEndRun() const { return nPostModuleStreamEndRun_.load(); }

  unsigned int TestServiceOne::nPreGlobalBeginRun() const { return nPreGlobalBeginRun_.load(); }
  unsigned int TestServiceOne::nPostGlobalBeginRun() const { return nPostGlobalBeginRun_.load(); }
  unsigned int TestServiceOne::nPreGlobalEndRun() const { return nPreGlobalEndRun_.load(); }
  unsigned int TestServiceOne::nPostGlobalEndRun() const { return nPostGlobalEndRun_.load(); }

  unsigned int TestServiceOne::nPreModuleGlobalBeginRun() const { return nPreModuleGlobalBeginRun_.load(); }
  unsigned int TestServiceOne::nPostModuleGlobalBeginRun() const { return nPostModuleGlobalBeginRun_.load(); }
  unsigned int TestServiceOne::nPreModuleGlobalEndRun() const { return nPreModuleGlobalEndRun_.load(); }
  unsigned int TestServiceOne::nPostModuleGlobalEndRun() const { return nPostModuleGlobalEndRun_.load(); }

  unsigned int TestServiceOne::nPreGlobalWriteRun() const { return nPreGlobalWriteRun_.load(); }
  unsigned int TestServiceOne::nPostGlobalWriteRun() const { return nPostGlobalWriteRun_.load(); }
}  // namespace edmtest

using edmtest::TestServiceOne;
DEFINE_FWK_SERVICE(TestServiceOne);

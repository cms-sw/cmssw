// -*- C++ -*-
//
// Package:     Services
// Class  :     Tracer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Sep  8 14:17:58 EDT 2005
//

#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"

#include <iostream>
#include <vector>

#include <string>
#include <set>

namespace edm {
  class ConfigurationDescriptions;
  class GlobalContext;
  class HLTPathStatus;
  class LuminosityBlock;
  class ModuleCallingContext;
  class ModuleDescription;
  class PathContext;
  class PathsAndConsumesOfModulesBase;
  class ProcessContext;
  class Run;
  class StreamContext;

  namespace service {
    class Tracer {
    public:
      Tracer(const ParameterSet&, ActivityRegistry&);

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void preallocate(service::SystemBounds const&);

      void preBeginJob(PathsAndConsumesOfModulesBase const&, ProcessContext const&);
      void postBeginJob();
      void postEndJob();

      void preSourceEvent(StreamID);
      void postSourceEvent(StreamID);

      void preSourceLumi(LuminosityBlockIndex);
      void postSourceLumi(LuminosityBlockIndex);

      void preSourceRun(RunIndex);
      void postSourceRun(RunIndex);

      void preSourceProcessBlock();
      void postSourceProcessBlock(std::string const&);

      void preOpenFile(std::string const&);
      void postOpenFile(std::string const&);

      void preCloseFile(std::string const& lfn);
      void postCloseFile(std::string const&);

      void preModuleBeginStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleBeginStream(StreamContext const&, ModuleCallingContext const&);

      void preModuleEndStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleEndStream(StreamContext const&, ModuleCallingContext const&);

      void preBeginProcessBlock(GlobalContext const&);
      void postBeginProcessBlock(GlobalContext const&);

      void preAccessInputProcessBlock(GlobalContext const&);
      void postAccessInputProcessBlock(GlobalContext const&);

      void preEndProcessBlock(GlobalContext const&);
      void postEndProcessBlock(GlobalContext const&);

      void preWriteProcessBlock(GlobalContext const&);
      void postWriteProcessBlock(GlobalContext const&);

      void preGlobalBeginRun(GlobalContext const&);
      void postGlobalBeginRun(GlobalContext const&);

      void preGlobalEndRun(GlobalContext const&);
      void postGlobalEndRun(GlobalContext const&);

      void preGlobalWriteRun(GlobalContext const&);
      void postGlobalWriteRun(GlobalContext const&);

      void preStreamBeginRun(StreamContext const&);
      void postStreamBeginRun(StreamContext const&);

      void preStreamEndRun(StreamContext const&);
      void postStreamEndRun(StreamContext const&);

      void preGlobalBeginLumi(GlobalContext const&);
      void postGlobalBeginLumi(GlobalContext const&);

      void preGlobalEndLumi(GlobalContext const&);
      void postGlobalEndLumi(GlobalContext const&);

      void preGlobalWriteLumi(GlobalContext const&);
      void postGlobalWriteLumi(GlobalContext const&);

      void preStreamBeginLumi(StreamContext const&);
      void postStreamBeginLumi(StreamContext const&);

      void preStreamEndLumi(StreamContext const&);
      void postStreamEndLumi(StreamContext const&);

      void preEvent(StreamContext const&);
      void postEvent(StreamContext const&);

      void prePathEvent(StreamContext const&, PathContext const&);
      void postPathEvent(StreamContext const&, PathContext const&, HLTPathStatus const&);

      void preModuleConstruction(ModuleDescription const& md);
      void postModuleConstruction(ModuleDescription const& md);

      void preModuleDestruction(ModuleDescription const& md);
      void postModuleDestruction(ModuleDescription const& md);

      void preModuleBeginJob(ModuleDescription const& md);
      void postModuleBeginJob(ModuleDescription const& md);

      void preModuleEndJob(ModuleDescription const& md);
      void postModuleEndJob(ModuleDescription const& md);

      void preModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void preModuleEventAcquire(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventAcquire(StreamContext const&, ModuleCallingContext const&);
      void preModuleEventDelayedGet(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventDelayedGet(StreamContext const&, ModuleCallingContext const&);
      void preEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postEventReadFromSource(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamPrefetching(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamPrefetching(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
      void preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
      void preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);

      void preModuleBeginProcessBlock(GlobalContext const&, ModuleCallingContext const&);
      void postModuleBeginProcessBlock(GlobalContext const&, ModuleCallingContext const&);
      void preModuleAccessInputProcessBlock(GlobalContext const&, ModuleCallingContext const&);
      void postModuleAccessInputProcessBlock(GlobalContext const&, ModuleCallingContext const&);
      void preModuleEndProcessBlock(GlobalContext const&, ModuleCallingContext const&);
      void postModuleEndProcessBlock(GlobalContext const&, ModuleCallingContext const&);

      void preModuleGlobalPrefetching(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalPrefetching(GlobalContext const&, ModuleCallingContext const&);

      void preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
      void preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);

      void preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
      void preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);

      void preModuleWriteProcessBlock(GlobalContext const&, ModuleCallingContext const&);
      void postModuleWriteProcessBlock(GlobalContext const&, ModuleCallingContext const&);

      void preModuleWriteRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleWriteRun(GlobalContext const&, ModuleCallingContext const&);

      void preModuleWriteLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleWriteLumi(GlobalContext const&, ModuleCallingContext const&);

      void preSourceConstruction(ModuleDescription const& md);
      void postSourceConstruction(ModuleDescription const& md);

      void preESModulePrefetching(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&);
      void postESModulePrefetching(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&);
      void preESModule(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&);
      void postESModule(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&);

    private:
      std::string indention_;
      std::set<std::string> dumpContextForLabels_;
      bool dumpNonModuleContext_;
      bool dumpPathsAndConsumes_;
      bool printTimestamps_;
      bool dumpEventSetupInfo_;
    };
  }  // namespace service
}  // namespace edm

using namespace edm::service;

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

}  // namespace

Tracer::Tracer(ParameterSet const& iPS, ActivityRegistry& iRegistry)
    : indention_(iPS.getUntrackedParameter<std::string>("indention")),
      dumpContextForLabels_(),
      dumpNonModuleContext_(iPS.getUntrackedParameter<bool>("dumpNonModuleContext")),
      dumpPathsAndConsumes_(iPS.getUntrackedParameter<bool>("dumpPathsAndConsumes")),
      printTimestamps_(iPS.getUntrackedParameter<bool>("printTimestamps")),
      dumpEventSetupInfo_(iPS.getUntrackedParameter<bool>("dumpEventSetupInfo")) {
  for (std::string& label : iPS.getUntrackedParameter<std::vector<std::string>>("dumpContextForLabels"))
    dumpContextForLabels_.insert(std::move(label));

  iRegistry.watchPreallocate(this, &Tracer::preallocate);

  iRegistry.watchPreBeginJob(this, &Tracer::preBeginJob);
  iRegistry.watchPostBeginJob(this, &Tracer::postBeginJob);
  iRegistry.watchPostEndJob(this, &Tracer::postEndJob);

  iRegistry.watchPreSourceEvent(this, &Tracer::preSourceEvent);
  iRegistry.watchPostSourceEvent(this, &Tracer::postSourceEvent);

  iRegistry.watchPreSourceLumi(this, &Tracer::preSourceLumi);
  iRegistry.watchPostSourceLumi(this, &Tracer::postSourceLumi);

  iRegistry.watchPreSourceRun(this, &Tracer::preSourceRun);
  iRegistry.watchPostSourceRun(this, &Tracer::postSourceRun);

  iRegistry.watchPreSourceProcessBlock(this, &Tracer::preSourceProcessBlock);
  iRegistry.watchPostSourceProcessBlock(this, &Tracer::postSourceProcessBlock);

  iRegistry.watchPreOpenFile(this, &Tracer::preOpenFile);
  iRegistry.watchPostOpenFile(this, &Tracer::postOpenFile);

  iRegistry.watchPreCloseFile(this, &Tracer::preCloseFile);
  iRegistry.watchPostCloseFile(this, &Tracer::postCloseFile);

  iRegistry.watchPreModuleBeginStream(this, &Tracer::preModuleBeginStream);
  iRegistry.watchPostModuleBeginStream(this, &Tracer::postModuleBeginStream);

  iRegistry.watchPreModuleEndStream(this, &Tracer::preModuleEndStream);
  iRegistry.watchPostModuleEndStream(this, &Tracer::postModuleEndStream);

  iRegistry.watchPreBeginProcessBlock(this, &Tracer::preBeginProcessBlock);
  iRegistry.watchPostBeginProcessBlock(this, &Tracer::postBeginProcessBlock);

  iRegistry.watchPreAccessInputProcessBlock(this, &Tracer::preAccessInputProcessBlock);
  iRegistry.watchPostAccessInputProcessBlock(this, &Tracer::postAccessInputProcessBlock);

  iRegistry.watchPreEndProcessBlock(this, &Tracer::preEndProcessBlock);
  iRegistry.watchPostEndProcessBlock(this, &Tracer::postEndProcessBlock);

  iRegistry.watchPreWriteProcessBlock(this, &Tracer::preWriteProcessBlock);
  iRegistry.watchPostWriteProcessBlock(this, &Tracer::postWriteProcessBlock);

  iRegistry.watchPreGlobalBeginRun(this, &Tracer::preGlobalBeginRun);
  iRegistry.watchPostGlobalBeginRun(this, &Tracer::postGlobalBeginRun);

  iRegistry.watchPreGlobalEndRun(this, &Tracer::preGlobalEndRun);
  iRegistry.watchPostGlobalEndRun(this, &Tracer::postGlobalEndRun);

  iRegistry.watchPreGlobalWriteRun(this, &Tracer::preGlobalWriteRun);
  iRegistry.watchPostGlobalWriteRun(this, &Tracer::postGlobalWriteRun);

  iRegistry.watchPreStreamBeginRun(this, &Tracer::preStreamBeginRun);
  iRegistry.watchPostStreamBeginRun(this, &Tracer::postStreamBeginRun);

  iRegistry.watchPreStreamEndRun(this, &Tracer::preStreamEndRun);
  iRegistry.watchPostStreamEndRun(this, &Tracer::postStreamEndRun);

  iRegistry.watchPreGlobalBeginLumi(this, &Tracer::preGlobalBeginLumi);
  iRegistry.watchPostGlobalBeginLumi(this, &Tracer::postGlobalBeginLumi);

  iRegistry.watchPreGlobalEndLumi(this, &Tracer::preGlobalEndLumi);
  iRegistry.watchPostGlobalEndLumi(this, &Tracer::postGlobalEndLumi);

  iRegistry.watchPreGlobalWriteLumi(this, &Tracer::preGlobalWriteLumi);
  iRegistry.watchPostGlobalWriteLumi(this, &Tracer::postGlobalWriteLumi);

  iRegistry.watchPreStreamBeginLumi(this, &Tracer::preStreamBeginLumi);
  iRegistry.watchPostStreamBeginLumi(this, &Tracer::postStreamBeginLumi);

  iRegistry.watchPreStreamEndLumi(this, &Tracer::preStreamEndLumi);
  iRegistry.watchPostStreamEndLumi(this, &Tracer::postStreamEndLumi);

  iRegistry.watchPreEvent(this, &Tracer::preEvent);
  iRegistry.watchPostEvent(this, &Tracer::postEvent);

  iRegistry.watchPrePathEvent(this, &Tracer::prePathEvent);
  iRegistry.watchPostPathEvent(this, &Tracer::postPathEvent);

  iRegistry.watchPreModuleConstruction(this, &Tracer::preModuleConstruction);
  iRegistry.watchPostModuleConstruction(this, &Tracer::postModuleConstruction);

  iRegistry.watchPreModuleDestruction(this, &Tracer::preModuleDestruction);
  iRegistry.watchPostModuleDestruction(this, &Tracer::postModuleDestruction);

  iRegistry.watchPreModuleBeginJob(this, &Tracer::preModuleBeginJob);
  iRegistry.watchPostModuleBeginJob(this, &Tracer::postModuleBeginJob);

  iRegistry.watchPreModuleEndJob(this, &Tracer::preModuleEndJob);
  iRegistry.watchPostModuleEndJob(this, &Tracer::postModuleEndJob);

  iRegistry.watchPreModuleEventPrefetching(this, &Tracer::preModuleEventPrefetching);
  iRegistry.watchPostModuleEventPrefetching(this, &Tracer::postModuleEventPrefetching);
  iRegistry.watchPreModuleEvent(this, &Tracer::preModuleEvent);
  iRegistry.watchPostModuleEvent(this, &Tracer::postModuleEvent);
  iRegistry.watchPreModuleEventAcquire(this, &Tracer::preModuleEventAcquire);
  iRegistry.watchPostModuleEventAcquire(this, &Tracer::postModuleEventAcquire);
  iRegistry.watchPreModuleEventDelayedGet(this, &Tracer::preModuleEventDelayedGet);
  iRegistry.watchPostModuleEventDelayedGet(this, &Tracer::postModuleEventDelayedGet);
  iRegistry.watchPreEventReadFromSource(this, &Tracer::preEventReadFromSource);
  iRegistry.watchPostEventReadFromSource(this, &Tracer::postEventReadFromSource);

  iRegistry.watchPreModuleStreamPrefetching(this, &Tracer::preModuleStreamPrefetching);
  iRegistry.watchPostModuleStreamPrefetching(this, &Tracer::postModuleStreamPrefetching);

  iRegistry.watchPreModuleStreamBeginRun(this, &Tracer::preModuleStreamBeginRun);
  iRegistry.watchPostModuleStreamBeginRun(this, &Tracer::postModuleStreamBeginRun);
  iRegistry.watchPreModuleStreamEndRun(this, &Tracer::preModuleStreamEndRun);
  iRegistry.watchPostModuleStreamEndRun(this, &Tracer::postModuleStreamEndRun);

  iRegistry.watchPreModuleStreamBeginLumi(this, &Tracer::preModuleStreamBeginLumi);
  iRegistry.watchPostModuleStreamBeginLumi(this, &Tracer::postModuleStreamBeginLumi);
  iRegistry.watchPreModuleStreamEndLumi(this, &Tracer::preModuleStreamEndLumi);
  iRegistry.watchPostModuleStreamEndLumi(this, &Tracer::postModuleStreamEndLumi);

  iRegistry.watchPreModuleBeginProcessBlock(this, &Tracer::preModuleBeginProcessBlock);
  iRegistry.watchPostModuleBeginProcessBlock(this, &Tracer::postModuleBeginProcessBlock);
  iRegistry.watchPreModuleAccessInputProcessBlock(this, &Tracer::preModuleAccessInputProcessBlock);
  iRegistry.watchPostModuleAccessInputProcessBlock(this, &Tracer::postModuleAccessInputProcessBlock);
  iRegistry.watchPreModuleEndProcessBlock(this, &Tracer::preModuleEndProcessBlock);
  iRegistry.watchPostModuleEndProcessBlock(this, &Tracer::postModuleEndProcessBlock);

  iRegistry.watchPreModuleGlobalPrefetching(this, &Tracer::preModuleGlobalPrefetching);
  iRegistry.watchPostModuleGlobalPrefetching(this, &Tracer::postModuleGlobalPrefetching);

  iRegistry.watchPreModuleGlobalBeginRun(this, &Tracer::preModuleGlobalBeginRun);
  iRegistry.watchPostModuleGlobalBeginRun(this, &Tracer::postModuleGlobalBeginRun);
  iRegistry.watchPreModuleGlobalEndRun(this, &Tracer::preModuleGlobalEndRun);
  iRegistry.watchPostModuleGlobalEndRun(this, &Tracer::postModuleGlobalEndRun);

  iRegistry.watchPreModuleGlobalBeginLumi(this, &Tracer::preModuleGlobalBeginLumi);
  iRegistry.watchPostModuleGlobalBeginLumi(this, &Tracer::postModuleGlobalBeginLumi);
  iRegistry.watchPreModuleGlobalEndLumi(this, &Tracer::preModuleGlobalEndLumi);
  iRegistry.watchPostModuleGlobalEndLumi(this, &Tracer::postModuleGlobalEndLumi);

  iRegistry.watchPreModuleWriteProcessBlock(this, &Tracer::preModuleWriteProcessBlock);
  iRegistry.watchPostModuleWriteProcessBlock(this, &Tracer::postModuleWriteProcessBlock);

  iRegistry.watchPreModuleWriteRun(this, &Tracer::preModuleWriteRun);
  iRegistry.watchPostModuleWriteRun(this, &Tracer::postModuleWriteRun);

  iRegistry.watchPreModuleWriteLumi(this, &Tracer::preModuleWriteLumi);
  iRegistry.watchPostModuleWriteLumi(this, &Tracer::postModuleWriteLumi);

  iRegistry.watchPreSourceConstruction(this, &Tracer::preSourceConstruction);
  iRegistry.watchPostSourceConstruction(this, &Tracer::postSourceConstruction);

  iRegistry.watchPreESModulePrefetching(this, &Tracer::preESModulePrefetching);
  iRegistry.watchPostESModulePrefetching(this, &Tracer::postESModulePrefetching);
  iRegistry.watchPreESModule(this, &Tracer::preESModule);
  iRegistry.watchPostESModule(this, &Tracer::postESModule);

  iRegistry.preSourceEarlyTerminationSignal_.connect([this](edm::TerminationOrigin iOrigin) {
    LogAbsolute out("Tracer");
    out << TimeStamper(printTimestamps_);
    out << indention_ << indention_ << " early termination before processing transition";
  });
  iRegistry.preStreamEarlyTerminationSignal_.connect(
      [this](edm::StreamContext const& iContext, edm::TerminationOrigin iOrigin) {
        LogAbsolute out("Tracer");
        out << TimeStamper(printTimestamps_);
        if (iContext.eventID().luminosityBlock() == 0) {
          out << indention_ << indention_ << " early termination of run: stream = " << iContext.streamID()
              << " run = " << iContext.eventID().run();
        } else {
          if (iContext.eventID().event() == 0) {
            out << indention_ << indention_ << " early termination of stream lumi: stream = " << iContext.streamID()
                << " run = " << iContext.eventID().run() << " lumi = " << iContext.eventID().luminosityBlock();
          } else {
            out << indention_ << indention_ << " early termination of event: stream = " << iContext.streamID()
                << " run = " << iContext.eventID().run() << " lumi = " << iContext.eventID().luminosityBlock()
                << " event = " << iContext.eventID().event();
          }
        }
        out << " : time = " << iContext.timestamp().value();
      });
  iRegistry.preGlobalEarlyTerminationSignal_.connect(
      [this](edm::GlobalContext const& iContext, edm::TerminationOrigin iOrigin) {
        LogAbsolute out("Tracer");
        out << TimeStamper(printTimestamps_);
        if (iContext.luminosityBlockID().value() == 0) {
          out << indention_ << indention_ << " early termination of global run " << iContext.luminosityBlockID().run();
        } else {
          out << indention_ << indention_
              << " early termination of global lumi run = " << iContext.luminosityBlockID().run()
              << " lumi = " << iContext.luminosityBlockID().luminosityBlock();
        }
        out << " : time = " << iContext.timestamp().value();
      });

  iRegistry.esSyncIOVQueuingSignal_.connect([this](edm::IOVSyncValue const& iSync) {
    LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_
                          << " queuing: EventSetup synchronization " << iSync.eventID();
  });
  iRegistry.preESSyncIOVSignal_.connect([this](edm::IOVSyncValue const& iSync) {
    LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_
                          << " pre: EventSetup synchronizing " << iSync.eventID();
  });
  iRegistry.postESSyncIOVSignal_.connect([this](edm::IOVSyncValue const& iSync) {
    LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_
                          << " post: EventSetup synchronizing " << iSync.eventID();
  });
}

void Tracer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("indention", "++")
      ->setComment("Prefix characters for output. The characters are repeated to form the indentation.");
  desc.addUntracked<std::vector<std::string>>("dumpContextForLabels", std::vector<std::string>{})
      ->setComment(
          "Prints context information to cout for the module transitions associated with these modules' labels");
  desc.addUntracked<bool>("dumpNonModuleContext", false)
      ->setComment("Prints context information to cout for the transitions not associated with any module label");
  desc.addUntracked<bool>("dumpPathsAndConsumes", false)
      ->setComment(
          "Prints information to cout about paths, endpaths, products consumed by modules and the dependencies between "
          "modules created by the products they consume");
  desc.addUntracked<bool>("printTimestamps", false)->setComment("Prints a time stamp for every transition");
  desc.addUntracked<bool>("dumpEventSetupInfo", false)
      ->setComment(
          "Prints info 3 times when an event setup cache is filled, before the lock, after the lock, and after "
          "filling");
  descriptions.add("Tracer", desc);
  descriptions.setComment(
      "This service prints each phase the framework is processing, e.g. constructing a module,running a module, etc.");
}

void Tracer::preallocate(service::SystemBounds const& bounds) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_
                        << " preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
                        << bounds.maxNumberOfConcurrentLuminosityBlocks() << " concurrent luminosity sections, "
                        << bounds.maxNumberOfStreams() << " streams";
}

void Tracer::preBeginJob(PathsAndConsumesOfModulesBase const& pathsAndConsumes, ProcessContext const& pc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_) << indention_ << " starting: begin job";
  if (dumpPathsAndConsumes_) {
    out << "\n"
        << "Process name = " << pc.processName() << "\n";
    out << "paths:\n";
    std::vector<std::string> const& paths = pathsAndConsumes.paths();
    for (auto const& path : paths) {
      out << "  " << path << "\n";
    }
    out << "end paths:\n";
    std::vector<std::string> const& endpaths = pathsAndConsumes.endPaths();
    for (auto const& endpath : endpaths) {
      out << "  " << endpath << "\n";
    }
    for (unsigned int j = 0; j < paths.size(); ++j) {
      std::vector<ModuleDescription const*> const& modulesOnPath = pathsAndConsumes.modulesOnPath(j);
      out << "modules on path " << paths.at(j) << ":\n";
      for (auto const& desc : modulesOnPath) {
        out << "  " << desc->moduleLabel() << "\n";
      }
    }
    for (unsigned int j = 0; j < endpaths.size(); ++j) {
      std::vector<ModuleDescription const*> const& modulesOnEndPath = pathsAndConsumes.modulesOnEndPath(j);
      out << "modules on end path " << endpaths.at(j) << ":\n";
      for (auto const& desc : modulesOnEndPath) {
        out << "  " << desc->moduleLabel() << "\n";
      }
    }
    std::vector<ModuleDescription const*> const& allModules = pathsAndConsumes.allModules();
    out << "All modules and modules in the current process whose products they consume:\n";
    out << "(This does not include modules from previous processes or the source)\n";
    out << "(Exclusively considers Event products, not Run, Lumi, or ProcessBlock products)\n";
    for (auto const& module : allModules) {
      out << "  " << module->moduleName() << "/\'" << module->moduleLabel() << "\'";
      unsigned int moduleID = module->id();
      if (pathsAndConsumes.moduleDescription(moduleID) != module) {
        throw cms::Exception("TestFailure") << "Tracer::preBeginJob, moduleDescription returns incorrect value";
      }
      std::vector<ModuleDescription const*> const& modulesWhoseProductsAreConsumedBy =
          pathsAndConsumes.modulesWhoseProductsAreConsumedBy(moduleID);
      if (!modulesWhoseProductsAreConsumedBy.empty()) {
        out << " consumes products from these modules:\n";
        for (auto const& producingModule : modulesWhoseProductsAreConsumedBy) {
          out << "    " << producingModule->moduleName() << "/\'" << producingModule->moduleLabel() << "\'\n";
        }
      } else {
        out << "\n";
      }
    }
    out << "All modules (listed by class and label) and all their consumed products.\n";
    out << "Consumed products are listed by type, label, instance, process.\n";
    out << "For products not in the event, \'processBlock\', \'run\' or \'lumi\' is added to indicate the TTree they "
           "are from.\n";
    out << "For products that are declared with mayConsume, \'may consume\' is added.\n";
    out << "For products consumed for Views, \'element type\' is added\n";
    out << "For products only read from previous processes, \'skip current process\' is added\n";
    for (auto const* module : allModules) {
      out << "  " << module->moduleName() << "/\'" << module->moduleLabel() << "\'";
      std::vector<ConsumesInfo> consumesInfo = pathsAndConsumes.consumesInfo(module->id());
      if (!consumesInfo.empty()) {
        out << " consumes:\n";
        for (auto const& info : consumesInfo) {
          out << "    " << info.type() << " \'" << info.label() << "\' \'" << info.instance();
          out << "\' \'" << info.process() << "\'";
          if (info.branchType() == InLumi) {
            out << ", lumi";
          } else if (info.branchType() == InRun) {
            out << ", run";
          } else if (info.branchType() == InProcess) {
            out << ", processBlock";
          }
          if (!info.alwaysGets()) {
            out << ", may consume";
          }
          if (info.kindOfType() == ELEMENT_TYPE) {
            out << ", element type";
          }
          if (info.skipCurrentProcess()) {
            out << ", skip current process";
          }
          out << "\n";
        }
      } else {
        out << "\n";
      }
    }
  }
}

void Tracer::postBeginJob() {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << " finished: begin job";
}

void Tracer::postEndJob() {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << " finished: end job";
}

void Tracer::preSourceEvent(StreamID sid) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_ << " starting: source event";
}

void Tracer::postSourceEvent(StreamID sid) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_ << " finished: source event";
}

void Tracer::preSourceLumi(LuminosityBlockIndex index) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_ << " starting: source lumi";
}

void Tracer::postSourceLumi(LuminosityBlockIndex index) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_ << " finished: source lumi";
}

void Tracer::preSourceRun(RunIndex index) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_ << " starting: source run";
}

void Tracer::postSourceRun(RunIndex index) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_ << " finished: source run";
}

void Tracer::preSourceProcessBlock() {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_
                        << " starting: source process block";
}

void Tracer::postSourceProcessBlock(std::string const& processName) {
  LogAbsolute("Tracer") << TimeStamper(printTimestamps_) << indention_ << indention_
                        << " finished: source process block " << processName;
}

void Tracer::preOpenFile(std::string const& lfn) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: open input file: lfn = " << lfn;
}

void Tracer::postOpenFile(std::string const& lfn) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: open input file: lfn = " << lfn;
}

void Tracer::preCloseFile(std::string const& lfn) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: close input file: lfn = " << lfn;
}
void Tracer::postCloseFile(std::string const& lfn) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: close input file: lfn = " << lfn;
}

void Tracer::preModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: begin stream for module: stream = " << sc.streamID() << " label = '"
      << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: begin stream for module: stream = " << sc.streamID() << " label = '"
      << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: end stream for module: stream = " << sc.streamID() << " label = '"
      << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = *mcc.moduleDescription();
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: end stream for module: stream = " << sc.streamID() << " label = '"
      << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preBeginProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: begin process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postBeginProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: begin process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preAccessInputProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: access input process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postAccessInputProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: access input process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preEndProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: end process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postEndProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: end process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preWriteProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " starting: write process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postWriteProcessBlock(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << indention_ << indention_ << " finished: write process block";
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preGlobalBeginRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: global begin run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postGlobalBeginRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: global begin run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preGlobalEndRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: global end run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postGlobalEndRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: global end run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preGlobalWriteRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: global write run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postGlobalWriteRun(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: global write run " << gc.luminosityBlockID().run()
      << " : time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preStreamBeginRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: begin run: stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::postStreamBeginRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: begin run: stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::preStreamEndRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: end run: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::postStreamEndRun(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: end run: stream = " << sc.streamID() << " run = " << sc.eventID().run()
      << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::preGlobalBeginLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: global begin lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postGlobalBeginLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: global begin lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preGlobalEndLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: global end lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postGlobalEndLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: global end lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preGlobalWriteLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: global write lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::postGlobalWriteLumi(GlobalContext const& gc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: global write lumi: run = " << gc.luminosityBlockID().run()
      << " lumi = " << gc.luminosityBlockID().luminosityBlock() << " time = " << gc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << gc;
  }
}

void Tracer::preStreamBeginLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: begin lumi: stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock()
      << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::postStreamBeginLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: begin lumi: stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock()
      << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::preStreamEndLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: end lumi: stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock()
      << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::postStreamEndLumi(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: end lumi: stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock()
      << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::preEvent(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: processing event : stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock()
      << " event = " << sc.eventID().event() << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::postEvent(StreamContext const& sc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: processing event : stream = " << sc.streamID()
      << " run = " << sc.eventID().run() << " lumi = " << sc.eventID().luminosityBlock()
      << " event = " << sc.eventID().event() << " time = " << sc.timestamp().value();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
  }
}

void Tracer::prePathEvent(StreamContext const& sc, PathContext const& pc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << indention_ << " starting: processing path '" << pc.pathName()
      << "' : stream = " << sc.streamID();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
    out << pc;
  }
}

void Tracer::postPathEvent(StreamContext const& sc, PathContext const& pc, HLTPathStatus const& hlts) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << indention_ << " finished: processing path '" << pc.pathName()
      << "' : stream = " << sc.streamID();
  if (dumpNonModuleContext_) {
    out << "\n" << sc;
    out << pc;
  }
}

void Tracer::preModuleConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: constructing module with label '" << desc.moduleLabel()
      << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::postModuleConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: constructing module with label '" << desc.moduleLabel()
      << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::preModuleDestruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " starting: destructing module with label '" << desc.moduleLabel()
      << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::postModuleDestruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_ << " finished: destructing module with label '" << desc.moduleLabel()
      << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::preModuleBeginJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_;
  out << " starting: begin job for module with label '" << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::postModuleBeginJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_;
  out << " finished: begin job for module with label '" << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::preModuleEndJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_;
  out << " starting: end job for module with label '" << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::postModuleEndJob(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_ << indention_;
  out << " finished: end job for module with label '" << desc.moduleLabel() << "' id = " << desc.id();
  if (dumpContextForLabels_.find(desc.moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << desc;
  }
}

void Tracer::preModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: prefetching before processing event for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: prefetching before processing event for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: processing event for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: processing event for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleEventAcquire(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: processing event acquire for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
}

void Tracer::postModuleEventAcquire(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: processing event acquire for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
}

void Tracer::preModuleEventDelayedGet(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: delayed processing event for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleEventDelayedGet(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: delayed processing event for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 5;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: event delayed read from source: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
}

void Tracer::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 5;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: event delayed read from source: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
}

void Tracer::preModuleStreamPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: prefetching before processing " << transitionName(sc.transition())
      << " for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleStreamPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: prefetching before processing " << transitionName(sc.transition())
      << " for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: begin run for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: begin run for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: end run for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: end run for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: begin lumi for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: begin lumi for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: end lumi for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: end lumi for module: stream = " << sc.streamID() << " label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << sc;
    out << mcc;
  }
}

void Tracer::preModuleGlobalPrefetching(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: prefetching before processing " << transitionName(gc.transition()) << " for module: label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleGlobalPrefetching(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: prefetching before processing " << transitionName(gc.transition()) << " for module: label = '"
      << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleBeginProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: begin process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleBeginProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: begin process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleAccessInputProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: access input process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleAccessInputProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: access input process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleEndProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: end process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleEndProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: end process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global begin run for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global begin run for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global end run for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global end run for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global begin lumi for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global begin lumi for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: global end lumi for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: global end lumi for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleWriteProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: write process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleWriteProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: write process block for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleWriteRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: write run for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleWriteRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: write run for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preModuleWriteLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: write lumi for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::postModuleWriteLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 3;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: write lumi for module: label = '" << mcc.moduleDescription()->moduleLabel()
      << "' id = " << mcc.moduleDescription()->id();
  if (dumpContextForLabels_.find(mcc.moduleDescription()->moduleLabel()) != dumpContextForLabels_.end()) {
    out << "\n" << gc;
    out << mcc;
  }
}

void Tracer::preSourceConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_;
  out << " starting: constructing source: " << desc.moduleName();
  if (dumpNonModuleContext_) {
    out << "\n" << desc;
  }
}

void Tracer::postSourceConstruction(ModuleDescription const& desc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  out << indention_;
  out << " finished: constructing source: " << desc.moduleName();
  if (dumpNonModuleContext_) {
    out << "\n" << desc;
  }
}

void Tracer::preESModulePrefetching(eventsetup::EventSetupRecordKey const& iKey, ESModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: prefetching for esmodule: label = '" << mcc.componentDescription()->label_
      << "' type = " << mcc.componentDescription()->type_ << " in record = " << iKey.name();
}

void Tracer::postESModulePrefetching(eventsetup::EventSetupRecordKey const& iKey, ESModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: prefetching for esmodule: label = '" << mcc.componentDescription()->label_
      << "' type = " << mcc.componentDescription()->type_ << " in record = " << iKey.name();
}

void Tracer::preESModule(eventsetup::EventSetupRecordKey const& iKey, ESModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " starting: processing esmodule: label = '" << mcc.componentDescription()->label_
      << "' type = " << mcc.componentDescription()->type_ << " in record = " << iKey.name();
}

void Tracer::postESModule(eventsetup::EventSetupRecordKey const& iKey, ESModuleCallingContext const& mcc) {
  LogAbsolute out("Tracer");
  out << TimeStamper(printTimestamps_);
  unsigned int nIndents = mcc.depth() + 4;
  for (unsigned int i = 0; i < nIndents; ++i) {
    out << indention_;
  }
  out << " finished: processing esmodule: label = '" << mcc.componentDescription()->label_
      << "' type = " << mcc.componentDescription()->type_ << " in record = " << iKey.name();
}

using edm::service::Tracer;
DEFINE_FWK_SERVICE(Tracer);

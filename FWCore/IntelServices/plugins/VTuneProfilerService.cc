// -*- C++ -*-
//
// Package:     Services
// Class  :     VTuneProfilerService

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
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
using namespace std::string_literals;

#include <boost/format.hpp>

#include <ittnotify.h>

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
    class VTuneProfilerService {
    public:
      VTuneProfilerService(const ParameterSet&, ActivityRegistry&);
      ~VTuneProfilerService();

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      void pre_module_check(std::string const & label);
      void post_module_check(std::string const & label);

      void preallocate(service::SystemBounds const&);

      void preBeginJob(PathsAndConsumesOfModulesBase const&, ProcessContext const&);
      void postBeginJob();
      void postEndJob();

      void preSourceEvent(StreamID);
      void postSourceEvent(StreamID);

      void preSourceLumi();
      void postSourceLumi();

      void preSourceRun();
      void postSourceRun();

      void preOpenFile(std::string const&, bool);
      void postOpenFile(std::string const&, bool);

      void preCloseFile(std::string const& lfn, bool primary);
      void postCloseFile(std::string const&, bool);

      void preModuleBeginStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleBeginStream(StreamContext const&, ModuleCallingContext const&);

      void preModuleEndStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleEndStream(StreamContext const&, ModuleCallingContext const&);

      void preGlobalBeginRun(GlobalContext const&);
      void postGlobalBeginRun(GlobalContext const&);

      void preGlobalEndRun(GlobalContext const&);
      void postGlobalEndRun(GlobalContext const&);

      void preStreamBeginRun(StreamContext const&);
      void postStreamBeginRun(StreamContext const&);

      void preStreamEndRun(StreamContext const&);
      void postStreamEndRun(StreamContext const&);

      void preGlobalBeginLumi(GlobalContext const&);
      void postGlobalBeginLumi(GlobalContext const&);

      void preGlobalEndLumi(GlobalContext const&);
      void postGlobalEndLumi(GlobalContext const&);

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

      void preModuleBeginJob(ModuleDescription const& md);
      void postModuleBeginJob(ModuleDescription const& md);

      void preModuleEndJob(ModuleDescription const& md);
      void postModuleEndJob(ModuleDescription const& md);

      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void preModuleEventDelayedGet(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventDelayedGet(StreamContext const&, ModuleCallingContext const&);
      void preEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postEventReadFromSource(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
      void preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
      void preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);

      void preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
      void preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);

      void preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
      void preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);

      void preSourceConstruction(ModuleDescription const& md);
      void postSourceConstruction(ModuleDescription const& md);

    private:
      std::vector<std::string>          m_enabledModules;
      bool                              m_showDelayedModules;

      __itt_domain *                    m_globalDomain;
      std::vector<__itt_domain *>       m_streamDomain;
      std::vector<__itt_event>          m_events;

    };
  }
}

using namespace edm::service;

VTuneProfilerService::VTuneProfilerService(ParameterSet const & iPS, ActivityRegistry & iRegistry) :
  m_enabledModules(iPS.getUntrackedParameter<std::vector<std::string>>("enableOnlyModules")),
  m_showDelayedModules(iPS.getUntrackedParameter<bool>("showDelayedModules")),
  m_globalDomain( __itt_domain_create("global") ),
  m_streamDomain()
{
  std::sort(m_enabledModules.begin(), m_enabledModules.end());
  
  if (m_globalDomain == nullptr) {
      // VTune Amplifier is not running
  } else {
    iRegistry.watchPreallocate(this, &VTuneProfilerService::preallocate);

    iRegistry.watchPreBeginJob(this, &VTuneProfilerService::preBeginJob);
    iRegistry.watchPostBeginJob(this, &VTuneProfilerService::postBeginJob);
    iRegistry.watchPostEndJob(this, &VTuneProfilerService::postEndJob);

    iRegistry.watchPreSourceEvent(this, &VTuneProfilerService::preSourceEvent);
    iRegistry.watchPostSourceEvent(this, &VTuneProfilerService::postSourceEvent);

    iRegistry.watchPreSourceLumi(this, &VTuneProfilerService::preSourceLumi);
    iRegistry.watchPostSourceLumi(this, &VTuneProfilerService::postSourceLumi);

    iRegistry.watchPreSourceRun(this, &VTuneProfilerService::preSourceRun);
    iRegistry.watchPostSourceRun(this, &VTuneProfilerService::postSourceRun);

    iRegistry.watchPreOpenFile(this, &VTuneProfilerService::preOpenFile);
    iRegistry.watchPostOpenFile(this, &VTuneProfilerService::postOpenFile);

    iRegistry.watchPreCloseFile(this, &VTuneProfilerService::preCloseFile);
    iRegistry.watchPostCloseFile(this, &VTuneProfilerService::postCloseFile);

    iRegistry.watchPreModuleBeginStream(this, &VTuneProfilerService::preModuleBeginStream);
    iRegistry.watchPostModuleBeginStream(this, &VTuneProfilerService::postModuleBeginStream);
    iRegistry.watchPreModuleEndStream(this, &VTuneProfilerService::preModuleEndStream);
    iRegistry.watchPostModuleEndStream(this, &VTuneProfilerService::postModuleEndStream);

    iRegistry.watchPreGlobalBeginRun(this, &VTuneProfilerService::preGlobalBeginRun);
    iRegistry.watchPostGlobalBeginRun(this, &VTuneProfilerService::postGlobalBeginRun);
    iRegistry.watchPreGlobalEndRun(this, &VTuneProfilerService::preGlobalEndRun);
    iRegistry.watchPostGlobalEndRun(this, &VTuneProfilerService::postGlobalEndRun);

    iRegistry.watchPreStreamBeginRun(this, &VTuneProfilerService::preStreamBeginRun);
    iRegistry.watchPostStreamBeginRun(this, &VTuneProfilerService::postStreamBeginRun);
    iRegistry.watchPreStreamEndRun(this, &VTuneProfilerService::preStreamEndRun);
    iRegistry.watchPostStreamEndRun(this, &VTuneProfilerService::postStreamEndRun);

    iRegistry.watchPreGlobalBeginLumi(this, &VTuneProfilerService::preGlobalBeginLumi);
    iRegistry.watchPostGlobalBeginLumi(this, &VTuneProfilerService::postGlobalBeginLumi);
    iRegistry.watchPreGlobalEndLumi(this, &VTuneProfilerService::preGlobalEndLumi);
    iRegistry.watchPostGlobalEndLumi(this, &VTuneProfilerService::postGlobalEndLumi);

    iRegistry.watchPreStreamBeginLumi(this, &VTuneProfilerService::preStreamBeginLumi);
    iRegistry.watchPostStreamBeginLumi(this, &VTuneProfilerService::postStreamBeginLumi);
    iRegistry.watchPreStreamEndLumi(this, &VTuneProfilerService::preStreamEndLumi);
    iRegistry.watchPostStreamEndLumi(this, &VTuneProfilerService::postStreamEndLumi);

    iRegistry.watchPreEvent(this, &VTuneProfilerService::preEvent);
    iRegistry.watchPostEvent(this, &VTuneProfilerService::postEvent);

    iRegistry.watchPrePathEvent(this, &VTuneProfilerService::prePathEvent);
    iRegistry.watchPostPathEvent(this, &VTuneProfilerService::postPathEvent);

    iRegistry.watchPreModuleConstruction(this, &VTuneProfilerService::preModuleConstruction);
    iRegistry.watchPostModuleConstruction(this, &VTuneProfilerService::postModuleConstruction);

    iRegistry.watchPreModuleBeginJob(this, &VTuneProfilerService::preModuleBeginJob);
    iRegistry.watchPostModuleBeginJob(this, &VTuneProfilerService::postModuleBeginJob);
    iRegistry.watchPreModuleEndJob(this, &VTuneProfilerService::preModuleEndJob);
    iRegistry.watchPostModuleEndJob(this, &VTuneProfilerService::postModuleEndJob);

    iRegistry.watchPreModuleEvent(this, &VTuneProfilerService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneProfilerService::postModuleEvent);
    if (m_showDelayedModules) {
      iRegistry.watchPreModuleEventDelayedGet(this, &VTuneProfilerService::preModuleEventDelayedGet);
      iRegistry.watchPostModuleEventDelayedGet(this, &VTuneProfilerService::postModuleEventDelayedGet);
    }
    iRegistry.watchPreEventReadFromSource(this, &VTuneProfilerService::preEventReadFromSource);
    iRegistry.watchPostEventReadFromSource(this, &VTuneProfilerService::postEventReadFromSource);

    iRegistry.watchPreModuleStreamBeginRun(this, &VTuneProfilerService::preModuleStreamBeginRun);
    iRegistry.watchPostModuleStreamBeginRun(this, &VTuneProfilerService::postModuleStreamBeginRun);
    iRegistry.watchPreModuleStreamEndRun(this, &VTuneProfilerService::preModuleStreamEndRun);
    iRegistry.watchPostModuleStreamEndRun(this, &VTuneProfilerService::postModuleStreamEndRun);

    iRegistry.watchPreModuleStreamBeginLumi(this, &VTuneProfilerService::preModuleStreamBeginLumi);
    iRegistry.watchPostModuleStreamBeginLumi(this, &VTuneProfilerService::postModuleStreamBeginLumi);
    iRegistry.watchPreModuleStreamEndLumi(this, &VTuneProfilerService::preModuleStreamEndLumi);
    iRegistry.watchPostModuleStreamEndLumi(this, &VTuneProfilerService::postModuleStreamEndLumi);

    iRegistry.watchPreModuleGlobalBeginRun(this, &VTuneProfilerService::preModuleGlobalBeginRun);
    iRegistry.watchPostModuleGlobalBeginRun(this, &VTuneProfilerService::postModuleGlobalBeginRun);
    iRegistry.watchPreModuleGlobalEndRun(this, &VTuneProfilerService::preModuleGlobalEndRun);
    iRegistry.watchPostModuleGlobalEndRun(this, &VTuneProfilerService::postModuleGlobalEndRun);

    iRegistry.watchPreModuleGlobalBeginLumi(this, &VTuneProfilerService::preModuleGlobalBeginLumi);
    iRegistry.watchPostModuleGlobalBeginLumi(this, &VTuneProfilerService::postModuleGlobalBeginLumi);
    iRegistry.watchPreModuleGlobalEndLumi(this, &VTuneProfilerService::preModuleGlobalEndLumi);
    iRegistry.watchPostModuleGlobalEndLumi(this, &VTuneProfilerService::postModuleGlobalEndLumi);

    iRegistry.watchPreSourceConstruction(this, &VTuneProfilerService::preSourceConstruction);
    iRegistry.watchPostSourceConstruction(this, &VTuneProfilerService::postSourceConstruction);

    // if only a subset of the modules should be profiled, start with VTune in paused mode
    if (m_enabledModules.empty())
      __itt_resume();
    else
      __itt_pause();
  }
}

VTuneProfilerService::~VTuneProfilerService() {
}

void
VTuneProfilerService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("enableOnlyModules", {})->setComment("");
  desc.addUntracked<bool>("showDelayedModules", true)->setComment("");
  descriptions.add("VTuneProfilerService", desc);
  descriptions.setComment("This Service provides CMSSW-aware annotations to nvprof/nvvm.");
}

void
VTuneProfilerService::pre_module_check(std::string const & label) {
  // if we are profiling only a subset of the modules, and this is one of them, resume the profiler
  if (not m_enabledModules.empty() and std::binary_search(m_enabledModules.begin(), m_enabledModules.end(), label))
    __itt_resume();
}

void
VTuneProfilerService::post_module_check(std::string const & label) {
  // if we are profiling only a subset of the modules, and this is one of them, resume the profiler
  if (not m_enabledModules.empty() and std::binary_search(m_enabledModules.begin(), m_enabledModules.end(), label))
    __itt_pause();
}

void
VTuneProfilerService::preallocate(service::SystemBounds const& bounds) {
  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
                         << bounds.maxNumberOfConcurrentLuminosityBlocks() << " concurrent luminosity sections, "
                         << bounds.maxNumberOfStreams() << " streams";
  m_streamDomain.resize(bounds.maxNumberOfStreams());
  m_events.resize(bounds.maxNumberOfStreams());
  // FIXME
  // warn if only some modules are enambled and there are multiple threads
  for (unsigned int i = 0; i < bounds.maxNumberOfStreams(); ++i)
    m_streamDomain[i] = __itt_domain_create((boost::format("stream %d") % i).str().c_str());
  auto label = out.str();
  auto event = __itt_event_create(label.c_str(), label.size());
  __itt_event_start(event);
}

void
VTuneProfilerService::preBeginJob(PathsAndConsumesOfModulesBase const& pathsAndConsumes, ProcessContext const& pc) {
  auto label = "preBeginJob";
  auto event = __itt_event_create(label, std::strlen(label));
  __itt_event_start(event);
}

void
VTuneProfilerService::postBeginJob() {
  auto label = "postBeginJob";
  auto event = __itt_event_create(label, std::strlen(label));
  __itt_event_start(event);
}

void
VTuneProfilerService::postEndJob() {
  auto label = "postEndJob";
  auto event = __itt_event_create(label, std::strlen(label));
  __itt_event_start(event);
}

void
VTuneProfilerService::preSourceEvent(StreamID sid) {
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create("preSourceEvent"));
}

void
VTuneProfilerService::postSourceEvent(StreamID sid) {
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preSourceLumi() {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create("preSourceLumi"));
}

void
VTuneProfilerService::postSourceLumi() {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preSourceRun() {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create("preSourceRun"));
}

void
VTuneProfilerService::postSourceRun() {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preOpenFile(std::string const& lfn, bool b) {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(("preOpenFile: "s + lfn).c_str()));
}

void
VTuneProfilerService::postOpenFile (std::string const& lfn, bool b) {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preCloseFile(std::string const & lfn, bool b) {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(("preCloseFile: "s + lfn).c_str()));
}

void
VTuneProfilerService::postCloseFile (std::string const& lfn, bool b) {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preGlobalBeginRun(GlobalContext const& gc) {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create("preGlobalBeginRun"));
}

void
VTuneProfilerService::postGlobalBeginRun(GlobalContext const& gc) {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preGlobalEndRun(GlobalContext const& gc) {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create("preGlobalEndRun"));
}

void
VTuneProfilerService::postGlobalEndRun(GlobalContext const& gc) {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preStreamBeginRun(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create("preStreamBeginRun"));
}

void
VTuneProfilerService::postStreamBeginRun(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preStreamEndRun(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create("preStreamEndRun"));
}

void
VTuneProfilerService::postStreamEndRun(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preGlobalBeginLumi(GlobalContext const& gc) {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create("preGlobalBeginLumi"));
}

void
VTuneProfilerService::postGlobalBeginLumi(GlobalContext const& gc) {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preGlobalEndLumi(GlobalContext const& gc) {
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create("preGlobalEndLumi"));
}

void
VTuneProfilerService::postGlobalEndLumi(GlobalContext const& gc) {
  __itt_task_end(m_globalDomain);
}

void
VTuneProfilerService::preStreamBeginLumi(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create("preStreamBeginLumi"));
}

void
VTuneProfilerService::postStreamBeginLumi(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preStreamEndLumi(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create("preStreamEndLumi"));
}

void
VTuneProfilerService::postStreamEndLumi(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preEvent(StreamContext const& sc) {
  auto sid = sc.streamID();
  auto label = (boost::format("event %d") % sc.eventID().event()).str();
  m_events[sid] = __itt_event_create(label.c_str(), label.size());
  __itt_event_start(m_events[sid]);

}

void
VTuneProfilerService::postEvent(StreamContext const& sc) {
  auto sid = sc.streamID();
  __itt_event_end(m_events[sid]);
}

void
VTuneProfilerService::prePathEvent(StreamContext const& sc, PathContext const& pc) {
  auto sid = sc.streamID();
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(("path "s + pc.pathName()).c_str()));
}

void
VTuneProfilerService::postPathEvent(StreamContext const& sc, PathContext const& pc, HLTPathStatus const& hlts) {
  auto sid = sc.streamID();
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preModuleConstruction(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleConstruction(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleBeginJob(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleBeginJob(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleEndJob(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleEndJob(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleEventDelayedGet(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(mcc.moduleDescription()->moduleLabel().c_str()));
}

void
VTuneProfilerService::postModuleEventDelayedGet(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  __itt_task_end(m_streamDomain[sid]);
}

void
VTuneProfilerService::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_streamDomain[sid], __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_streamDomain[sid]);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const & label = mcc.moduleDescription()->moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

void
VTuneProfilerService::preSourceConstruction(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  pre_module_check(label);
  __itt_task_begin(m_globalDomain, __itt_null, __itt_null, __itt_string_handle_create(label.c_str()));
}

void
VTuneProfilerService::postSourceConstruction(ModuleDescription const& desc) {
  auto const & label = desc.moduleLabel();
  __itt_task_end(m_globalDomain);
  post_module_check(label);
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using edm::service::VTuneProfilerService;
DEFINE_FWK_SERVICE(VTuneProfilerService);

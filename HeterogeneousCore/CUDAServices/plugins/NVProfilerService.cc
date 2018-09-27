// -*- C++ -*-
//
// Package:     HeterogeneousCore/CUDAServices
// Class  :     NVProfilerService

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include <boost/format.hpp>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

using namespace std::string_literals;

namespace {
  int nvtxDomainRangePush(nvtxDomainHandle_t domain, const char* message) {
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version         = NVTX_VERSION;
    eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii   = message;
    return nvtxDomainRangePushEx(domain, &eventAttrib);
  }

  __attribute__((unused))
  int nvtxDomainRangePushColor(nvtxDomainHandle_t domain, const char* message, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version         = NVTX_VERSION;
    eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType       = NVTX_COLOR_ARGB;
    eventAttrib.color           = color;
    eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii   = message;
    return nvtxDomainRangePushEx(domain, &eventAttrib);
  }

  __attribute__((unused))
  nvtxRangeId_t nvtxDomainRangeStart(nvtxDomainHandle_t domain, const char* message) {
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version         = NVTX_VERSION;
    eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii   = message;
    return nvtxDomainRangeStartEx(domain, &eventAttrib);
  }

  nvtxRangeId_t nvtxDomainRangeStartColor(nvtxDomainHandle_t domain, const char* message, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version         = NVTX_VERSION;
    eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType       = NVTX_COLOR_ARGB;
    eventAttrib.color           = color;
    eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii   = message;
    return nvtxDomainRangeStartEx(domain, &eventAttrib);
  }

  void nvtxDomainMark(nvtxDomainHandle_t domain, const char* message) {
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version         = NVTX_VERSION;
    eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii   = message;
    nvtxDomainMarkEx(domain, &eventAttrib);
  }

  __attribute__((unused))
  void nvtxDomainMarkColor(nvtxDomainHandle_t domain, const char* message, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version         = NVTX_VERSION;
    eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType       = NVTX_COLOR_ARGB;
    eventAttrib.color           = color;
    eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii   = message;
    nvtxDomainMarkEx(domain, &eventAttrib);
  }

  enum {
    nvtxBlack       = 0x00000000,
    nvtxRed         = 0x00ff0000,
    nvtxDarkGreen   = 0x00009900,
    nvtxGreen       = 0x0000ff00,
    nvtxLightGreen  = 0x00ccffcc,
    nvtxBlue        = 0x000000ff,
    nvtxAmber       = 0x00ffbf00,
    nvtxLightAmber  = 0x00fff2cc,
    nvtxWhite       = 0x00ffffff
  };

  constexpr nvtxRangeId_t nvtxInvalidRangeId = 0xfffffffffffffffful;
}

class NVProfilerService {
public:
  NVProfilerService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~NVProfilerService();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void preallocate(edm::service::SystemBounds const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);
  void postBeginJob();

  // there is no preEndJob() signal
  void postEndJob();

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalBeginRun(edm::GlobalContext const&);
  void postGlobalBeginRun(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalEndRun(edm::GlobalContext const&);
  void postGlobalEndRun(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamBeginRun(edm::StreamContext const&);
  void postStreamBeginRun(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamEndRun(edm::StreamContext const&);
  void postStreamEndRun(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalBeginLumi(edm::GlobalContext const&);
  void postGlobalBeginLumi(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalEndLumi(edm::GlobalContext const&);
  void postGlobalEndLumi(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamBeginLumi(edm::StreamContext const&);
  void postStreamBeginLumi(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamEndLumi(edm::StreamContext const&);
  void postStreamEndLumi(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preEvent(edm::StreamContext const&);
  void postEvent(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void prePathEvent(edm::StreamContext const&, edm::PathContext const&);
  void postPathEvent(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preOpenFile(std::string const&, bool);
  void postOpenFile(std::string const&, bool);

  // these signal pair are guaranteed to be called by the same thread
  void preCloseFile(std::string const&, bool);
  void postCloseFile(std::string const&, bool);

  // these signal pair are guaranteed to be called by the same thread
  void preSourceConstruction(edm::ModuleDescription const&);
  void postSourceConstruction(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preSourceRun(edm::RunIndex);
  void postSourceRun(edm::RunIndex);

  // these signal pair are guaranteed to be called by the same thread
  void preSourceLumi(edm::LuminosityBlockIndex);
  void postSourceLumi(edm::LuminosityBlockIndex);

  // these signal pair are guaranteed to be called by the same thread
  void preSourceEvent(edm::StreamID);
  void postSourceEvent(edm::StreamID);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleConstruction(edm::ModuleDescription const&);
  void postModuleConstruction(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleBeginJob(edm::ModuleDescription const&);
  void postModuleBeginJob(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEndJob(edm::ModuleDescription const&);
  void postModuleEndJob(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEndStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEndStream(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEventAcquire(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventAcquire(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEventDelayedGet(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventDelayedGet(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&);

private:
  bool highlight(std::string const& label) const {
    return (std::binary_search(highlightModules_.begin(), highlightModules_.end(), label));
  }

  uint32_t labelColor(std::string const& label) const {
    return highlight(label) ? nvtxAmber : nvtxGreen;
  }

  uint32_t labelColorLight(std::string const& label) const {
    return highlight(label) ? nvtxLightAmber : nvtxLightGreen;
  }

  std::vector<std::string>                  highlightModules_;
  const bool                                showModulePrefetching_;
  bool                                      skipFirstEvent_;

  unsigned int                              concurrentStreams_;
  bool                                      globalFirstEventDone_ = false;
  std::vector<bool>                         streamFirstEventDone_;
  std::vector<nvtxRangeId_t>                event_;                 // per-stream event ranges
  std::vector<std::vector<nvtxRangeId_t>>   stream_modules_;        // per-stream, per-module ranges
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<nvtxRangeId_t>     global_modules_;        // global per-module events

private:
  struct Domains {
    nvtxDomainHandle_t              global;
    std::vector<nvtxDomainHandle_t> stream;

    Domains(NVProfilerService* service) {
      global = nvtxDomainCreate("EDM Global");
      allocate_streams(service->concurrentStreams_);
    }

    ~Domains() {
      nvtxDomainDestroy(global);
      for (unsigned int sid = 0; sid < stream.size(); ++sid) {
        nvtxDomainDestroy(stream[sid]);
      }
    }

    void allocate_streams(unsigned int streams) {
      stream.resize(streams);
      for (unsigned int sid = 0; sid < streams; ++sid) {
        stream[sid] = nvtxDomainCreate((boost::format("EDM Stream %d") % sid).str().c_str());
      }
    }
  };

  // allow access to concurrentStreams_
  friend struct Domains;

  tbb::enumerable_thread_specific<Domains> domains_;

  nvtxDomainHandle_t global_domain() {
    return domains_.local().global;
  }

  nvtxDomainHandle_t stream_domain(unsigned int sid) {
    return domains_.local().stream.at(sid);
  }

};

NVProfilerService::NVProfilerService(edm::ParameterSet const & config, edm::ActivityRegistry & registry) :
  highlightModules_(config.getUntrackedParameter<std::vector<std::string>>("highlightModules")),
  showModulePrefetching_(config.getUntrackedParameter<bool>("showModulePrefetching")),
  skipFirstEvent_(config.getUntrackedParameter<bool>("skipFirstEvent")),
  concurrentStreams_(0),
  domains_(this)
{
  // make sure that CUDA is initialised, and that the CUDAService destructor is called after this service's destructor
  edm::Service<CUDAService> cudaService;

  std::sort(highlightModules_.begin(), highlightModules_.end());

  // enables profile collection; if profiling is already enabled it has no effect
  if (not skipFirstEvent_) {
    cudaProfilerStart();
  }

  registry.watchPreallocate(this, &NVProfilerService::preallocate);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreBeginJob(this, &NVProfilerService::preBeginJob);
  registry.watchPostBeginJob(this, &NVProfilerService::postBeginJob);

  // there is no preEndJob() signal
  registry.watchPostEndJob(this, &NVProfilerService::postEndJob);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalBeginRun(this, &NVProfilerService::preGlobalBeginRun);
  registry.watchPostGlobalBeginRun(this, &NVProfilerService::postGlobalBeginRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalEndRun(this, &NVProfilerService::preGlobalEndRun);
  registry.watchPostGlobalEndRun(this, &NVProfilerService::postGlobalEndRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamBeginRun(this, &NVProfilerService::preStreamBeginRun);
  registry.watchPostStreamBeginRun(this, &NVProfilerService::postStreamBeginRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamEndRun(this, &NVProfilerService::preStreamEndRun);
  registry.watchPostStreamEndRun(this, &NVProfilerService::postStreamEndRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalBeginLumi(this, &NVProfilerService::preGlobalBeginLumi);
  registry.watchPostGlobalBeginLumi(this, &NVProfilerService::postGlobalBeginLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalEndLumi(this, &NVProfilerService::preGlobalEndLumi);
  registry.watchPostGlobalEndLumi(this, &NVProfilerService::postGlobalEndLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamBeginLumi(this, &NVProfilerService::preStreamBeginLumi);
  registry.watchPostStreamBeginLumi(this, &NVProfilerService::postStreamBeginLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamEndLumi(this, &NVProfilerService::preStreamEndLumi);
  registry.watchPostStreamEndLumi(this, &NVProfilerService::postStreamEndLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreEvent(this, &NVProfilerService::preEvent);
  registry.watchPostEvent(this, &NVProfilerService::postEvent);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPrePathEvent(this, &NVProfilerService::prePathEvent);
  registry.watchPostPathEvent(this, &NVProfilerService::postPathEvent);

  if (showModulePrefetching_) {
    // these signal pair are NOT guaranteed to be called by the same thread
    registry.watchPreModuleEventPrefetching(this, &NVProfilerService::preModuleEventPrefetching);
    registry.watchPostModuleEventPrefetching(this, &NVProfilerService::postModuleEventPrefetching);
  }

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreOpenFile(this, &NVProfilerService::preOpenFile);
  registry.watchPostOpenFile(this, &NVProfilerService::postOpenFile);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreCloseFile(this, &NVProfilerService::preCloseFile);
  registry.watchPostCloseFile(this, &NVProfilerService::postCloseFile);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceConstruction(this, &NVProfilerService::preSourceConstruction);
  registry.watchPostSourceConstruction(this, &NVProfilerService::postSourceConstruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceRun(this, &NVProfilerService::preSourceRun);
  registry.watchPostSourceRun(this, &NVProfilerService::postSourceRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceLumi(this, &NVProfilerService::preSourceLumi);
  registry.watchPostSourceLumi(this, &NVProfilerService::postSourceLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceEvent(this, &NVProfilerService::preSourceEvent);
  registry.watchPostSourceEvent(this, &NVProfilerService::postSourceEvent);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleConstruction(this, &NVProfilerService::preModuleConstruction);
  registry.watchPostModuleConstruction(this, &NVProfilerService::postModuleConstruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleBeginJob(this, &NVProfilerService::preModuleBeginJob);
  registry.watchPostModuleBeginJob(this, &NVProfilerService::postModuleBeginJob);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleEndJob(this, &NVProfilerService::preModuleEndJob);
  registry.watchPostModuleEndJob(this, &NVProfilerService::postModuleEndJob);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleBeginStream(this, &NVProfilerService::preModuleBeginStream);
  registry.watchPostModuleBeginStream(this, &NVProfilerService::postModuleBeginStream);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleEndStream(this, &NVProfilerService::preModuleEndStream);
  registry.watchPostModuleEndStream(this, &NVProfilerService::postModuleEndStream);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalBeginRun(this, &NVProfilerService::preModuleGlobalBeginRun);
  registry.watchPostModuleGlobalBeginRun(this, &NVProfilerService::postModuleGlobalBeginRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalEndRun(this, &NVProfilerService::preModuleGlobalEndRun);
  registry.watchPostModuleGlobalEndRun(this, &NVProfilerService::postModuleGlobalEndRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalBeginLumi(this, &NVProfilerService::preModuleGlobalBeginLumi);
  registry.watchPostModuleGlobalBeginLumi(this, &NVProfilerService::postModuleGlobalBeginLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalEndLumi(this, &NVProfilerService::preModuleGlobalEndLumi);
  registry.watchPostModuleGlobalEndLumi(this, &NVProfilerService::postModuleGlobalEndLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleStreamBeginRun(this, &NVProfilerService::preModuleStreamBeginRun);
  registry.watchPostModuleStreamBeginRun(this, &NVProfilerService::postModuleStreamBeginRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleStreamEndRun(this, &NVProfilerService::preModuleStreamEndRun);
  registry.watchPostModuleStreamEndRun(this, &NVProfilerService::postModuleStreamEndRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleStreamBeginLumi(this, &NVProfilerService::preModuleStreamBeginLumi);
  registry.watchPostModuleStreamBeginLumi(this, &NVProfilerService::postModuleStreamBeginLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleStreamEndLumi(this, &NVProfilerService::preModuleStreamEndLumi);
  registry.watchPostModuleStreamEndLumi(this, &NVProfilerService::postModuleStreamEndLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleEventAcquire(this, &NVProfilerService::preModuleEventAcquire);
  registry.watchPostModuleEventAcquire(this, &NVProfilerService::postModuleEventAcquire);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleEvent(this, &NVProfilerService::preModuleEvent);
  registry.watchPostModuleEvent(this, &NVProfilerService::postModuleEvent);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleEventDelayedGet(this, &NVProfilerService::preModuleEventDelayedGet);
  registry.watchPostModuleEventDelayedGet(this, &NVProfilerService::postModuleEventDelayedGet);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreEventReadFromSource(this, &NVProfilerService::preEventReadFromSource);
  registry.watchPostEventReadFromSource(this, &NVProfilerService::postEventReadFromSource);
}

NVProfilerService::~NVProfilerService() {
  cudaProfilerStop();
}

void
NVProfilerService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("highlightModules", {})->setComment("");
  desc.addUntracked<bool>("showModulePrefetching", false)->setComment("Show the stack of dependencies that requested to run a module.");
  desc.addUntracked<bool>("skipFirstEvent", false)->setComment("Start profiling after the first event has completed.\nWith multiple streams, ignore transitions belonging to events started in parallel to the first event.\nRequires running nvprof with the '--profile-from-start off' option.");
  descriptions.add("NVProfilerService", desc);
  descriptions.setComment(R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

Notes on nvprof options:
  - the option '--profile-from-start off' should be used if skipFirstEvent is True.
  - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
  - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)");
}

void
NVProfilerService::preallocate(edm::service::SystemBounds const& bounds) {
  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
                         << bounds.maxNumberOfConcurrentLuminosityBlocks() << " luminosity sections, "
                         << bounds.maxNumberOfStreams() << " streams\nrunning on"
                         << bounds.maxNumberOfThreads() << " threads";
  nvtxDomainMark(global_domain(), out.str().c_str());

  concurrentStreams_ = bounds.maxNumberOfStreams();
  for (auto& domain: domains_) {
    domain.allocate_streams(concurrentStreams_);
  }
  event_.resize(concurrentStreams_);
  stream_modules_.resize(concurrentStreams_);
  if (skipFirstEvent_) {
    globalFirstEventDone_ = false;
    streamFirstEventDone_.resize(concurrentStreams_, false);
  }
}

void
NVProfilerService::preBeginJob(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes, edm::ProcessContext const& pc) {
  nvtxDomainMark(global_domain(), "preBeginJob");

  // FIXME this probably works only in the absence of subprocesses
  // size() + 1 because pathsAndConsumes.allModules() does not include the source
  unsigned int modules = pathsAndConsumes.allModules().size() + 1;
  global_modules_.resize(modules, nvtxInvalidRangeId);
  for (unsigned int sid = 0; sid < concurrentStreams_; ++sid) {
    stream_modules_[sid].resize(modules, nvtxInvalidRangeId);
  }
}

void
NVProfilerService::postBeginJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainMark(global_domain(), "postBeginJob");
  }
}

void
NVProfilerService::postEndJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainMark(global_domain(), "postEndJob");
  }
}

void
NVProfilerService::preSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePush(stream_domain(sid), "source");
  }
}

void
NVProfilerService::postSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePop(stream_domain(sid));
  }
}

void
NVProfilerService::preSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), "source lumi");
  }
}

void
NVProfilerService::postSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), "source run");
  }
}

void
NVProfilerService::postSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preOpenFile(std::string const& lfn, bool) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), ("open file "s + lfn).c_str());
  }
}

void
NVProfilerService::postOpenFile(std::string const& lfn, bool) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preCloseFile(std::string const & lfn, bool) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), ("close file "s + lfn).c_str());
  }
}

void
NVProfilerService::postCloseFile(std::string const& lfn, bool) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preModuleBeginStream(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " begin stream";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleBeginStream(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleEndStream(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " end stream";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleEndStream(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), "global begin run");
  }
}

void
NVProfilerService::postGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), "global end run");
  }
}

void
NVProfilerService::postGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePush(stream_domain(sid), "stream begin run");
  }
}

void
NVProfilerService::postStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePop(stream_domain(sid));
  }
}

void
NVProfilerService::preStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePush(stream_domain(sid), "stream end run");
  }
}

void
NVProfilerService::postStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePop(stream_domain(sid));
  }
}

void
NVProfilerService::preGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), "global begin lumi");
  }
}

void
NVProfilerService::postGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePush(global_domain(), "global end lumi");
  }
}

void
NVProfilerService::postGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    nvtxDomainRangePop(global_domain());
  }
}

void
NVProfilerService::preStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePush(stream_domain(sid), "stream begin lumi");
  }
}

void
NVProfilerService::postStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePop(stream_domain(sid));
  }
}

void
NVProfilerService::preStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  nvtxDomainRangePush(stream_domain(sid), "stream end lumi");
}

void
NVProfilerService::postStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangePop(stream_domain(sid));
  }
}

void
NVProfilerService::preEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid] = nvtxDomainRangeStartColor(stream_domain(sid), "event", nvtxDarkGreen);
  }
}

void
NVProfilerService::postEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainRangeEnd(stream_domain(sid), event_[sid]);
    event_[sid] = nvtxInvalidRangeId;
  } else {
    streamFirstEventDone_[sid] = true;
    // there is a possible race condition among different threads processing different events;
    // however, cudaProfilerStart() is supposed to be thread-safe and ignore multiple calls, so this should not be an issue.
    if (std::all_of(streamFirstEventDone_.begin(), streamFirstEventDone_.end(), [](bool x){ return x; })) {
      globalFirstEventDone_ = true;
      cudaProfilerStart();
    }
  }
}

void
NVProfilerService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainMark(global_domain(), ("before path "s + pc.pathName()).c_str());
  }
}

void
NVProfilerService::postPathEvent(edm::StreamContext const& sc, edm::PathContext const& pc, edm::HLTPathStatus const& hlts) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    nvtxDomainMark(global_domain(), ("after path "s + pc.pathName()).c_str());
  }
}

void
NVProfilerService::preModuleEventPrefetching(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " prefetching";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColorLight(label));
  }
}

void
NVProfilerService::postModuleEventPrefetching(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    global_modules_.grow_to_at_least(mid+1);
    auto const & label = desc.moduleLabel();
    auto const & msg = label + " construction";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const & label = desc.moduleLabel();
    auto const & msg = label + " begin job";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const & label = desc.moduleLabel();
    auto const & msg = label + " end job";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " acquire";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), label.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleEventDelayedGet(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  /* FIXME
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " delayed get";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), label.c_str(), labelColorLight(label));
  }
  */
}

void
NVProfilerService::postModuleEventDelayedGet(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  /* FIXME
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
  */
}

void
NVProfilerService::preEventReadFromSource(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  /* FIXME
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " read from source";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColorLight(label));
  }
  */
}

void
NVProfilerService::postEventReadFromSource(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  /* FIXME
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
  */
}

void
NVProfilerService::preModuleStreamBeginRun(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " stream begin run";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleStreamBeginRun(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleStreamEndRun(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " stream end run";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleStreamEndRun(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleStreamBeginLumi(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " stream begin lumi";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleStreamBeginLumi(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleStreamEndLumi(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " stream end lumi";
    assert(stream_modules_[sid][mid] == nvtxInvalidRangeId);
    stream_modules_[sid][mid] = nvtxDomainRangeStartColor(stream_domain(sid), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleStreamEndLumi(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(stream_domain(sid), stream_modules_[sid][mid]);
    stream_modules_[sid][mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleGlobalBeginRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " global begin run";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleGlobalBeginRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleGlobalEndRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " global end run";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleGlobalEndRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleGlobalBeginLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " global begin lumi";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleGlobalBeginLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preModuleGlobalEndLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const & label = mcc.moduleDescription()->moduleLabel();
    auto const & msg = label + " global end lumi";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postModuleGlobalEndLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

void
NVProfilerService::preSourceConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    global_modules_.grow_to_at_least(mid+1);
    auto const & label = desc.moduleLabel();
    auto const & msg = label + " construction";
    global_modules_[mid] = nvtxDomainRangeStartColor(global_domain(), msg.c_str(), labelColor(label));
  }
}

void
NVProfilerService::postSourceConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    nvtxDomainRangeEnd(global_domain(), global_modules_[mid]);
    global_modules_[mid] = nvtxInvalidRangeId;
  }
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(NVProfilerService);

#ifndef __FWCore_Services_ProfilerService_h__
#define __FWCore_Services_ProfilerService_h__

#include "FWCore/Services/interface/ProfilerServiceBase.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>

#include <fmt/printf.h>

#include <boost/stacktrace.hpp>

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
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

/**
 * Based template class for range/mark based profiling services, targeting
 * NVidia NVTX, AMP ROCmTX, or VTune.
 */

/**
 * Helper macros to declare signal handler pairs by parameter signature.
 */
#define DECLARE_SIGNAL_WATCHER_NOARGS(signal) \
  void pre##signal();                         \
  void post##signal();

#define DECLARE_SIGNAL_WATCHER_PROCESS_CONTEXT(signal) \
  void pre##signal(edm::ProcessContext const&);        \
  void post##signal();

#define DECLARE_SIGNAL_WATCHER_SOURCE_PROCESS_BLOCK(signal) \
  void pre##signal();                                       \
  void post##signal(std::string const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(signal) \
  void pre##signal(edm::StreamContext const&);        \
  void post##signal(edm::StreamContext const&);

#define DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(signal) \
  void pre##signal(edm::GlobalContext const&);        \
  void post##signal(edm::GlobalContext const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_ID(signal) \
  void pre##signal(edm::StreamID);               \
  void post##signal(edm::StreamID);

#define DECLARE_SIGNAL_WATCHER_LUMIBLOCK_INDEX(signal) \
  void pre##signal(edm::LuminosityBlockIndex);         \
  void post##signal(edm::LuminosityBlockIndex);

#define DECLARE_SIGNAL_WATCHER_RUN_INDEX(signal) \
  void pre##signal(edm::RunIndex);               \
  void post##signal(edm::RunIndex);

#define DECLARE_SIGNAL_WATCHER_STRING(signal) \
  void pre##signal(std::string const&);       \
  void post##signal(std::string const&);

#define DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(signal) \
  void pre##signal(edm::ModuleDescription const&);        \
  void post##signal(edm::ModuleDescription const&);

#define DECLARE_SIGNAL_WATCHER_COMPONENT_DESCRIPTION(signal)      \
  void pre##signal(edm::eventsetup::ComponentDescription const&); \
  void post##signal(edm::eventsetup::ComponentDescription const&);

#define DECLARE_SIGNAL_WATCHER_IOV_SYNC_VALUE(signal) \
  void pre##signal(edm::IOVSyncValue const&);         \
  void post##signal(edm::IOVSyncValue const&);

#define DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(signal)              \
  void pre##signal(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const&); \
  void post##signal(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT(signal) \
  void pre##signal(edm::StreamContext const&, edm::PathContext const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT_HLT_STATUS(signal) \
  void post##signal(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(signal)     \
  void pre##signal(edm::StreamContext const&, edm::ModuleCallingContext const&); \
  void post##signal(edm::StreamContext const&, edm::ModuleCallingContext const&);

#define DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(signal)     \
  void pre##signal(edm::GlobalContext const&, edm::ModuleCallingContext const&); \
  void post##signal(edm::GlobalContext const&, edm::ModuleCallingContext const&);

#define DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_STREAM(signal) \
  void pre##signal(edm::StreamContext const&, edm::TerminationOrigin);

#define DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_GLOBAL(signal) \
  void pre##signal(edm::GlobalContext const&, edm::TerminationOrigin);

#define DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_SOURCE(signal) void pre##signal(edm::TerminationOrigin);

// Useful for starting constructs using std::string::operator+() with a litteral string.
using namespace std::string_literals;

/**
 * @brief Base class for profiling services.
 * @tparam Backend The backend implementation class.
 * The backend will have to implement the actual range/mark operations, plus
 * capture and domains management.
 * Current expected classes and functions are:
 * - Range class with:
 *   - startColorIn(domain, message, color, func)
 *   - endIn(domain, message, func)
 * - markColorIn(domain, message, color, func)
 * - Domain management class with:
 *   - domainCreate(name)
 *   - domainDestroy(domain) (maybe destructor will be enough)
 * - Start/stop the underlying EDM service.
 * - profilerStart()
 * - profilerStop() (maybe wrapped into a class with the previous function)
 */
template <typename Backend>
class ProfilerService : public ProfilerServiceBase {
public:
  using Range = typename Backend::Range;
  using Domain = typename Backend::Domain;

  ProfilerService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~ProfilerService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /******** Infrastructure/setup signal pairs *************************************/

  void postServicesConstruction();

  void preEventSetupModulesConstruction();
  void postEventSetupModulesConstruction();

  void preModulesAndSourceConstruction();
  void postModulesAndSourceConstruction();

  void preFinishSchedule();
  void postFinishSchedule();

  void prePrincipalsCreation();
  void postPrincipalsCreation();

  void preScheduleConsistencyCheck();
  void postScheduleConsistencyCheck();

  void preallocate(edm::service::SystemBounds const&);

  void preEventSetupConfigurationFinalized();
  void postEventSetupConfigurationFinalized();

  void eventSetupConfiguration(edm::eventsetup::ESRecordsToProductResolverIndices const&, edm::ProcessContext const&);

  void preModulesInitializationFinalized();
  void postModulesInitializationFinalized();

  DECLARE_SIGNAL_WATCHER_PROCESS_CONTEXT(BeginJob)

  DECLARE_SIGNAL_WATCHER_NOARGS(EndJob)

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(BeginStream)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(EndStream)

  void jobFailure();

  /******** Source transition signals *********************************************/

  void preSourceNextTransition();
  void postSourceNextTransition();

  /******** Source stream context signals *************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_ID(SourceEvent)

  /******** Source lumi context signals *************************************/

  DECLARE_SIGNAL_WATCHER_LUMIBLOCK_INDEX(SourceLumi)

  /******** Source run context signals *************************************/

  DECLARE_SIGNAL_WATCHER_RUN_INDEX(SourceRun)

  /******** Source process block signals *************************************/

  DECLARE_SIGNAL_WATCHER_SOURCE_PROCESS_BLOCK(SourceProcessBlock)

  /******** File context signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STRING(OpenFile)
  DECLARE_SIGNAL_WATCHER_STRING(CloseFile)

  /******** Output file signals **********************************************/

  DECLARE_SIGNAL_WATCHER_NOARGS(OpenOutputFiles)
  DECLARE_SIGNAL_WATCHER_NOARGS(CloseOutputFiles)

  /******** Module stream context signals *********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleBeginStream)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEndStream)

  /******** Process block signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(BeginProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(AccessInputProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(EndProcessBlock)

  /******** Job-level single signals *********************************************/

  void beginProcessing();
  void endProcessing();

  /******* Global context signals  **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalBeginRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalEndRun)

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(WriteProcessBlock)

  /******** Global write signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalWriteRun)

  /******* Stream context signals  **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamBeginRun)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamEndRun)

  /******** Global context lumi signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalBeginLumi)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalEndLumi)

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalWriteLumi)

  /******** Stream context lumi signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamBeginLumi)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamEndLumi)

  /******** Stream context events signal **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(Event)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(ClearEvent)

  /******** Path context event signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT(PathEvent)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT_HLT_STATUS(PathEvent)

  /******** Early termination signals (Pre only, no Post) *****************************/

  DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_STREAM(StreamEarlyTermination)
  DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_GLOBAL(GlobalEarlyTermination)
  DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_SOURCE(SourceEarlyTermination)

  /******** ES module construction signals **********************************************/

  DECLARE_SIGNAL_WATCHER_COMPONENT_DESCRIPTION(ESModuleConstruction)

  /******** ES module context signals *********************************************/

  void postESModuleRegistration(edm::eventsetup::ComponentDescription const&);

  /******** ES IOV sync signals **********************************************/

  void esSyncIOVQueuing(edm::IOVSyncValue const&);

  DECLARE_SIGNAL_WATCHER_IOV_SYNC_VALUE(ESSyncIOV)

  // Prefetching is optionally watched
  // (see constructor)
  DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(ESModulePrefetching)
  DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(ESModule)
  DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(ESModuleAcquire)

  /******** Module no-context signals *********************************************/

  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleConstruction)
  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleDestruction)
  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleBeginJob)
  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleEndJob)

  /******** Module context signals *********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEventPrefetching)

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEvent)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEventAcquire)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleTransformPrefetching)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleTransform)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleTransformAcquiring)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEventDelayedGet)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(EventReadFromSource)

  /******** Module stream prefetching signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamPrefetching)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamBeginRun)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamEndRun)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamBeginLumi)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamEndLumi)

  /******** Module global/process block context signals *************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleBeginProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleAccessInputProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEndProcessBlock)

  /******** Module global prefetching and process block signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalPrefetching)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalBeginRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalEndRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalBeginLumi)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalEndLumi)

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleWriteProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleWriteRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleWriteLumi)

  /******** Source module context signals *************************************/

  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(SourceConstruction)
#undef DECLARE_SIGNAL_WATCHER_NOARGS
#undef DECLARE_SIGNAL_WATCHER_PROCESS_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_ID
#undef DECLARE_SIGNAL_WATCHER_LUMIBLOCK_INDEX
#undef DECLARE_SIGNAL_WATCHER_RUN_INDEX
#undef DECLARE_SIGNAL_WATCHER_STRING
#undef DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION
#undef DECLARE_SIGNAL_WATCHER_COMPONENT_DESCRIPTION
#undef DECLARE_SIGNAL_WATCHER_IOV_SYNC_VALUE
#undef DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT_HLT_STATUS
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_STREAM
#undef DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_GLOBAL
#undef DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_SOURCE

private:
  using SharedRangePool = ProfilerServiceBase::RangePool<Range>;
  using GlobalInFlightRanges = ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, std::string>;
  using GlobalESInFlightRanges = ProfilerServiceBase::
      InFlightRanges<Backend, Range, Domain, unsigned int, std::string, edm::ESModuleCallingContext::State, std::uintptr_t>;
  using StreamModuleInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int>;
  using TransformInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int, std::uintptr_t>;
  using IndexInFlightRanges = ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int>;
  using TwoIndexInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int>;
  using ThreeIndexInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int, unsigned int>;
  using PathInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int, bool>;

  bool highlight(std::string const& label) const {
    return (std::binary_search(highlightModules_.begin(), highlightModules_.end(), label));
  }

  // Highlight exception: if `label` names a highlighted module, remap `color` to the Amber family
  // while preserving its shade (Dark2..Light2); otherwise return `color` unchanged.
  Color highlightColor(Color color, std::string const& label) const {
    return highlight(label) ? to_highlighted(color) : color;
  }

  std::vector<std::string> highlightModules_;
  const bool showModulePrefetching_;
  const bool skipFirstEvent_;
  const bool showDetailedInfo_;

  std::atomic<bool> globalFirstEventDone_ = false;
  std::vector<std::atomic<bool>> streamFirstEventDone_;
  SharedRangePool range_pool_;
  GlobalInFlightRanges global_in_flight_ranges_;
  GlobalESInFlightRanges global_es_in_flight_ranges_;
  StreamModuleInFlightRanges stream_modules_in_flight_ranges_;
  StreamModuleInFlightRanges stream_modules_event_in_flight_ranges_;
  StreamModuleInFlightRanges stream_modules_event_acquire_in_flight_ranges_;
  TransformInFlightRanges transform_in_flight_ranges_;
  IndexInFlightRanges event_in_flight_ranges_;           // per-stream event ranges, keyed by stream id
  IndexInFlightRanges source_in_flight_ranges_;          // per-stream source ranges, keyed by stream id
  PathInFlightRanges path_in_flight_ranges_;             // per-stream, per-path ranges, keyed by (sid, pid, isEndPath)
  IndexInFlightRanges global_modules_in_flight_ranges_;  // global per-module ranges, keyed by module id
  IndexInFlightRanges global_ES_modules_in_flight_ranges_;  // global per-ES-module ranges, keyed by component id
  IndexInFlightRanges global_run_in_flight_ranges_;         // global per-run ranges, keyed by run number
  TwoIndexInFlightRanges global_lumi_in_flight_ranges_;     // global per-lumi ranges, keyed by (run, lumi)
  TwoIndexInFlightRanges stream_run_in_flight_ranges_;      // per-stream run ranges, keyed by (sid, run)
  ThreeIndexInFlightRanges stream_lumi_in_flight_ranges_;   // per-stream lumi ranges, keyed by (sid, run, lumi)

  Domain global_domain_;               // NVTX domain for global EDM transitions
  std::vector<Domain> stream_domain_;  // NVTX domains for per-EDM-stream transitions
};

// This macro registers signal watchers pairs. Same for all.
#define REGISTER_SIGNAL_WATCHER(signal)                           \
  registry.watchPre##signal(this, &ProfilerService::pre##signal); \
  registry.watchPost##signal(this, &ProfilerService::post##signal);

template <typename Backend>
ProfilerService<Backend>::ProfilerService(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
    : highlightModules_(config.getUntrackedParameter<std::vector<std::string>>("highlightModules")),
      showModulePrefetching_(config.getUntrackedParameter<bool>("showModulePrefetching")),
      skipFirstEvent_(config.getUntrackedParameter<bool>("skipFirstEvent")),
      showDetailedInfo_(config.getUntrackedParameter<bool>("showDetailedInfo")),
      range_pool_(),
      global_in_flight_ranges_(range_pool_, showDetailedInfo_),
      global_es_in_flight_ranges_(range_pool_, showDetailedInfo_),
      stream_modules_in_flight_ranges_(range_pool_, showDetailedInfo_),
      stream_modules_event_in_flight_ranges_(range_pool_, showDetailedInfo_),
      stream_modules_event_acquire_in_flight_ranges_(range_pool_, showDetailedInfo_),
      transform_in_flight_ranges_(range_pool_, showDetailedInfo_),
      event_in_flight_ranges_(range_pool_, showDetailedInfo_),
      source_in_flight_ranges_(range_pool_, showDetailedInfo_),
      path_in_flight_ranges_(range_pool_, showDetailedInfo_),
      global_modules_in_flight_ranges_(range_pool_, showDetailedInfo_),
      global_ES_modules_in_flight_ranges_(range_pool_, showDetailedInfo_),
      global_run_in_flight_ranges_(range_pool_, showDetailedInfo_),
      global_lumi_in_flight_ranges_(range_pool_, showDetailedInfo_),
      stream_run_in_flight_ranges_(range_pool_, showDetailedInfo_),
      stream_lumi_in_flight_ranges_(range_pool_, showDetailedInfo_) {
  // make sure that CUDA is initialised, and that the CUDAInterface destructor is called after this service's destructor
  typename Backend::EDMService service;
  std::cout << Backend::shortName() << "ProfilerService: initializing..." << std::endl;
  if (not service) {
    std::cout << Backend::shortName() << "ProfilerService: EDM service not available, disabling profiling service"
              << std::endl;
    return;
  }
  if (not service or not service->enabled()) {
    std::cout << Backend::shortName()
              << "ProfilerService: EDM service failed to be enabled, disabling profiling service" << std::endl;
    return;
  }
  std::cout << Backend::shortName()
            << "ProfilerService: EDM service initialized successfully. Registering watchers to EDM." << std::endl;

  std::sort(highlightModules_.begin(), highlightModules_.end());

  // create the NVTX domain for global EDM transitions
  global_domain_.create("EDM Global");

  // enables profile collection; if profiling is already enabled it has no effect
  // otherwise, make sure it is stopped.
  if (not skipFirstEvent_) {
    Backend::profilerStart();
  } else {
    Backend::profilerStop();
  }

  // Keep watcher registration order aligned with ActivityRegistry::watch* declarations.

  registry.watchPostServicesConstruction(this, &ProfilerService::postServicesConstruction);

  REGISTER_SIGNAL_WATCHER(EventSetupModulesConstruction)

  REGISTER_SIGNAL_WATCHER(ModulesAndSourceConstruction)

  REGISTER_SIGNAL_WATCHER(FinishSchedule)

  REGISTER_SIGNAL_WATCHER(PrincipalsCreation)

  REGISTER_SIGNAL_WATCHER(ScheduleConsistencyCheck)

  registry.watchPreallocate(this, &ProfilerService::preallocate);

  REGISTER_SIGNAL_WATCHER(EventSetupConfigurationFinalized)
  registry.watchEventSetupConfiguration(this, &ProfilerService::eventSetupConfiguration);

  REGISTER_SIGNAL_WATCHER(ModulesInitializationFinalized)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(BeginJob)
  REGISTER_SIGNAL_WATCHER(EndJob)

  registry.watchLookupInitializationComplete(this, &ProfilerService::lookupInitializationComplete);

  REGISTER_SIGNAL_WATCHER(BeginStream)
  REGISTER_SIGNAL_WATCHER(EndStream)

  registry.watchJobFailure(this, &ProfilerService::jobFailure);

  REGISTER_SIGNAL_WATCHER(SourceNextTransition)

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(SourceEvent)
  REGISTER_SIGNAL_WATCHER(SourceLumi)
  REGISTER_SIGNAL_WATCHER(SourceRun)
  REGISTER_SIGNAL_WATCHER(SourceProcessBlock)

  REGISTER_SIGNAL_WATCHER(OpenFile)
  REGISTER_SIGNAL_WATCHER(CloseFile)
  REGISTER_SIGNAL_WATCHER(OpenOutputFiles)
  REGISTER_SIGNAL_WATCHER(CloseOutputFiles)
  /******** Module stream context signals *********************************************/
  REGISTER_SIGNAL_WATCHER(ModuleBeginStream)
  REGISTER_SIGNAL_WATCHER(ModuleEndStream)

  // Process block signal pairs
  REGISTER_SIGNAL_WATCHER(BeginProcessBlock)
  REGISTER_SIGNAL_WATCHER(AccessInputProcessBlock)
  REGISTER_SIGNAL_WATCHER(EndProcessBlock)

  // Job-level single signals
  registry.watchBeginProcessing(this, &ProfilerService::beginProcessing);
  registry.watchEndProcessing(this, &ProfilerService::endProcessing);

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(GlobalBeginRun)
  REGISTER_SIGNAL_WATCHER(GlobalEndRun)

  REGISTER_SIGNAL_WATCHER(WriteProcessBlock)

  // Global write signal pairs
  REGISTER_SIGNAL_WATCHER(GlobalWriteRun)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(StreamBeginRun)
  REGISTER_SIGNAL_WATCHER(StreamEndRun)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(GlobalBeginLumi)
  REGISTER_SIGNAL_WATCHER(GlobalEndLumi)

  REGISTER_SIGNAL_WATCHER(GlobalWriteLumi)

  REGISTER_SIGNAL_WATCHER(StreamBeginLumi)
  REGISTER_SIGNAL_WATCHER(StreamEndLumi)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(Event)

  REGISTER_SIGNAL_WATCHER(ClearEvent)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(PathEvent)

  // Early termination signals (Pre only)
  registry.watchPreStreamEarlyTermination(this, &ProfilerService::preStreamEarlyTermination);
  registry.watchPreGlobalEarlyTermination(this, &ProfilerService::preGlobalEarlyTermination);
  registry.watchPreSourceEarlyTermination(this, &ProfilerService::preSourceEarlyTermination);

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(ESModuleConstruction)

  // ES signal watchers
  registry.watchPostESModuleRegistration(this, &ProfilerService::postESModuleRegistration);

  // ES IOV sync signals
  registry.watchESSyncIOVQueuing(this, &ProfilerService::esSyncIOVQueuing);
  REGISTER_SIGNAL_WATCHER(ESSyncIOV)

  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ESModulePrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ESModule)
  REGISTER_SIGNAL_WATCHER(ESModuleAcquire)

  REGISTER_SIGNAL_WATCHER(ModuleConstruction)
  REGISTER_SIGNAL_WATCHER(ModuleDestruction)

  REGISTER_SIGNAL_WATCHER(ModuleBeginJob)
  REGISTER_SIGNAL_WATCHER(ModuleEndJob)

  if (showModulePrefetching_) {
    // these signal pair are NOT guaranteed to be called by the same thread
    REGISTER_SIGNAL_WATCHER(ModuleEventPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleEvent)
  REGISTER_SIGNAL_WATCHER(ModuleEventAcquire)
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleTransformPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleTransform)
  REGISTER_SIGNAL_WATCHER(ModuleTransformAcquiring)
  REGISTER_SIGNAL_WATCHER(ModuleEventDelayedGet)
  REGISTER_SIGNAL_WATCHER(EventReadFromSource)

  // Module stream prefetching signal pair
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleStreamPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndLumi)

  REGISTER_SIGNAL_WATCHER(ModuleBeginProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleAccessInputProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleEndProcessBlock)

  // Module global prefetching and process block signal pairs
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleGlobalPrefetching)
  }

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(ModuleGlobalBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalEndLumi)

  REGISTER_SIGNAL_WATCHER(ModuleWriteProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleWriteRun)
  REGISTER_SIGNAL_WATCHER(ModuleWriteLumi)

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(SourceConstruction)
}

#undef REGISTER_SIGNAL_WATCHER

template <typename Backend>
ProfilerService<Backend>::~ProfilerService() {
  for (unsigned int sid = 0; sid < stream_domain_.size(); ++sid) {
    stream_domain_[sid].destroy();
  }
  global_domain_.destroy();
  Backend::profilerStop();
}

template <typename Backend>
void ProfilerService<Backend>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("highlightModules", {})->setComment("");
  desc.addUntracked<bool>("showModulePrefetching", false)
      ->setComment("Show the stack of dependencies that requested to run a module.");
  desc.addUntracked<bool>("skipFirstEvent", false)
      ->setComment(
          "Start profiling after the first event has completed.\nWith multiple streams, ignore transitions belonging "
          "to events started in parallel to the first event.\nRequires running nvprof with the '--profile-from-start "
          "off' option.");
  desc.addUntracked<bool>("showDetailedInfo", false)
      ->setComment(
          "Show values of the module, path, and event transitions parameters.\n"
          "When enabled, show many details in the profiler timeline.\n"
          "When disabled, enables per-module, etc... statistics.");
  descriptions.add(Backend::shortName() + "ProfilerService", desc);
  descriptions.setComment(Backend::serviceComment());
  // For reference, here is a possible extended comment for nvprof/nvvm backends:
  //   descriptions.setComment(R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

  // Notes on nvprof options:
  //   - the option '--profile-from-start off' should be used if skipFirstEvent is True.
  //   - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
  //   - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)");
}

/******** Signal-watcher implementation macros (expanded in the sections below) ********/

// ES module signal ranges are keyed dynamically to avoid collisions from overlapping calls.
#define DEFINE_ES_SIGNAL_WATCHER(signal, color)                                                         \
  template <class Backend>                                                                              \
  void ProfilerService<Backend>::pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey,          \
                                             edm::ESModuleCallingContext const& esmcc) {                \
    auto mid = esmcc.componentDescription()->id_;                                                       \
    auto const& label = esmcc.componentDescription()->label_;                                           \
    auto const& type = esmcc.componentDescription()->type_;                                             \
    auto const& state = esmcc.state();                                                                  \
    auto const callId = esmcc.callID();                                                                 \
    std::string detail = label.empty() ? (type + "(type)") : (label + " type=" + type);                 \
    global_es_in_flight_ranges_.start(global_domain_,                                                   \
                                      color,                                                            \
                                      __func__,                                                         \
                                      std::string(#signal) + " " + detail,                              \
                                      "",                                                               \
                                      "mid record state callId",                                        \
                                      mid,                                                              \
                                      iKey.name(),                                                      \
                                      state,                                                            \
                                      callId);                                                          \
  }                                                                                                     \
  template <class Backend>                                                                              \
  void ProfilerService<Backend>::post##signal(edm::eventsetup::EventSetupRecordKey const& iKey,         \
                                              edm::ESModuleCallingContext const& esmcc) {               \
    auto mid = esmcc.componentDescription()->id_;                                                       \
    auto const& state = esmcc.state();                                                                  \
    auto const callId = esmcc.callID();                                                                 \
    global_es_in_flight_ranges_.end(                                                                    \
        global_domain_, __func__, #signal, "mid record state callId", mid, iKey.name(), state, callId); \
  }

// Macro for per-stream module (StreamContext, ModuleCallingContext) signal pairs, keyed by (sid, mid).
#define DEFINE_MODULE_STREAM_SIGNAL_WATCHER(signal, inFlightRanges, color)                                          \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {  \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      std::string detail = label + " type=" + mcc.moduleDescription()->moduleName();                                \
      inFlightRanges.start(stream_domain_[sid],                                                                     \
                           highlightColor(color, label),                                                            \
                           __func__,                                                                                \
                           std::string(#signal) + " " + detail,                                                     \
                           "",                                                                                      \
                           "sid mid",                                                                               \
                           sid,                                                                                     \
                           mid);                                                                                    \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      inFlightRanges.end(stream_domain_[sid], __func__, #signal, "sid mid", sid, mid);                              \
    }                                                                                                               \
  }

// Macro for module transform (StreamContext, ModuleCallingContext) signal pairs, keyed by (sid, mid, callId).
#define DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(signal, color)                                                       \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {  \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const callId = mcc.callID();                                                                             \
      std::string detail = label + " type=" + mcc.moduleDescription()->moduleName();                                \
      transform_in_flight_ranges_.start(stream_domain_[sid],                                                        \
                                        highlightColor(color, label),                                               \
                                        __func__,                                                                   \
                                        std::string(#signal) + " " + detail,                                        \
                                        "",                                                                         \
                                        "sid mid callId",                                                           \
                                        sid,                                                                        \
                                        mid,                                                                        \
                                        callId);                                                                    \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const callId = mcc.callID();                                                                             \
      transform_in_flight_ranges_.end(stream_domain_[sid], __func__, #signal, "sid mid callId", sid, mid, callId);  \
    }                                                                                                               \
  }

// Macro for global-module (GlobalContext, ModuleCallingContext) signal pairs, keyed by module id.
#define DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(signal, color)                                                              \
  template <class Backend>                                                                                              \
  void ProfilerService<Backend>::pre##signal(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {      \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                                 \
      auto mid = mcc.moduleDescription()->id();                                                                         \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                       \
      std::string detail = label + " type=" + mcc.moduleDescription()->moduleName();                                    \
      global_modules_in_flight_ranges_.start(                                                                           \
          global_domain_, highlightColor(color, label), __func__, std::string(#signal) + " " + detail, "", "mid", mid); \
    }                                                                                                                   \
  }                                                                                                                     \
  template <class Backend>                                                                                              \
  void ProfilerService<Backend>::post##signal(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {     \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                                 \
      auto mid = mcc.moduleDescription()->id();                                                                         \
      global_modules_in_flight_ranges_.end(global_domain_, __func__, #signal, "mid", mid);                              \
    }                                                                                                                   \
  }

// Macro for module (ModuleDescription) signal pairs, keyed by module id. `guard` selects when to record.
#define DEFINE_MODULE_DESC_SIGNAL_WATCHER(signal, color, guard)                                                         \
  template <class Backend>                                                                                              \
  void ProfilerService<Backend>::pre##signal(edm::ModuleDescription const& desc) {                                      \
    if (guard) {                                                                                                        \
      auto mid = desc.id();                                                                                             \
      auto const& label = desc.moduleLabel();                                                                           \
      std::string detail = label + " type=" + desc.moduleName();                                                        \
      global_modules_in_flight_ranges_.start(                                                                           \
          global_domain_, highlightColor(color, label), __func__, std::string(#signal) + " " + detail, "", "mid", mid); \
    }                                                                                                                   \
  }                                                                                                                     \
  template <class Backend>                                                                                              \
  void ProfilerService<Backend>::post##signal(edm::ModuleDescription const& desc) {                                     \
    if (guard) {                                                                                                        \
      auto mid = desc.id();                                                                                             \
      global_modules_in_flight_ranges_.end(global_domain_, __func__, #signal, "mid", mid);                              \
    }                                                                                                                   \
  }

// Macro for ES module construction (ComponentDescription) signal pairs, keyed by component id.
#define DEFINE_ES_CONSTRUCTION_SIGNAL_WATCHER(signal, color)                                       \
  template <class Backend>                                                                         \
  void ProfilerService<Backend>::pre##signal(edm::eventsetup::ComponentDescription const& desc) {  \
    if (not skipFirstEvent_) {                                                                     \
      auto mid = desc.id_;                                                                         \
      auto const& label = desc.label_;                                                             \
      auto const& type = desc.type_;                                                               \
      std::string detail = label.empty() ? (type + "(type)") : (label + " type=" + type);          \
      global_ES_modules_in_flight_ranges_.start(                                                   \
          global_domain_, color, __func__, std::string(#signal) + " " + detail, "", "mid", mid);   \
    }                                                                                              \
  }                                                                                                \
  template <class Backend>                                                                         \
  void ProfilerService<Backend>::post##signal(edm::eventsetup::ComponentDescription const& desc) { \
    if (not skipFirstEvent_) {                                                                     \
      auto mid = desc.id_;                                                                         \
      global_ES_modules_in_flight_ranges_.end(global_domain_, __func__, #signal, "mid", mid);      \
    }                                                                                              \
  }

// Macro for per-stream (StreamContext) signal pairs, keyed by stream id via event_in_flight_ranges_.
#define DEFINE_STREAM_SIGNAL_WATCHER(signal, color)                                                 \
  template <class Backend>                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc) {                        \
    auto sid = sc.streamID();                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                        \
      event_in_flight_ranges_.start(stream_domain_[sid], color, __func__, #signal, "", "sid", sid); \
    }                                                                                               \
  }                                                                                                 \
  template <class Backend>                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc) {                       \
    auto sid = sc.streamID();                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                        \
      event_in_flight_ranges_.end(stream_domain_[sid], __func__, #signal, "sid", sid);              \
    }                                                                                               \
  }

// Macro for per-stream id (StreamID) signal pairs, keyed by stream id via source_in_flight_ranges_.
#define DEFINE_STREAM_ID_SIGNAL_WATCHER(signal, color)                                               \
  template <class Backend>                                                                           \
  void ProfilerService<Backend>::pre##signal(edm::StreamID sid) {                                    \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                         \
      source_in_flight_ranges_.start(stream_domain_[sid], color, __func__, #signal, "", "sid", sid); \
    }                                                                                                \
  }                                                                                                  \
  template <class Backend>                                                                           \
  void ProfilerService<Backend>::post##signal(edm::StreamID sid) {                                   \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                         \
      source_in_flight_ranges_.end(stream_domain_[sid], __func__, #signal, "sid", sid);              \
    }                                                                                                \
  }

// Macro for per-stream run (StreamContext) signal pairs, keyed by (sid, run).
#define DEFINE_STREAM_RUN_SIGNAL_WATCHER(signal, color)                                                               \
  template <class Backend>                                                                                            \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc) {                                          \
    auto sid = sc.streamID();                                                                                         \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                          \
      auto run = sc.eventID().run();                                                                                  \
      std::string detail = "runSlot=" + std::to_string(sc.runIndex().value());                                        \
      stream_run_in_flight_ranges_.start(stream_domain_[sid], color, __func__, #signal, detail, "sid run", sid, run); \
    }                                                                                                                 \
  }                                                                                                                   \
  template <class Backend>                                                                                            \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc) {                                         \
    auto sid = sc.streamID();                                                                                         \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                          \
      auto run = sc.eventID().run();                                                                                  \
      stream_run_in_flight_ranges_.end(stream_domain_[sid], __func__, #signal, "sid run", sid, run);                  \
    }                                                                                                                 \
  }

// Macro for per-stream lumi (StreamContext) signal pairs, keyed by (sid, run, lumi).
#define DEFINE_STREAM_LUMI_SIGNAL_WATCHER(signal, color)                                                         \
  template <class Backend>                                                                                       \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc) {                                     \
    auto sid = sc.streamID();                                                                                    \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                     \
      auto run = sc.eventID().run();                                                                             \
      auto lumi = sc.eventID().luminosityBlock();                                                                \
      std::string detail = "runSlot=" + std::to_string(sc.runIndex().value()) +                                  \
                           " lumiSlot=" + std::to_string(sc.luminosityBlockIndex().value());                     \
      stream_lumi_in_flight_ranges_.start(                                                                       \
          stream_domain_[sid], color, __func__, #signal, detail, "sid run lumi", sid, run, lumi);                \
    }                                                                                                            \
  }                                                                                                              \
  template <class Backend>                                                                                       \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc) {                                    \
    auto sid = sc.streamID();                                                                                    \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                     \
      auto run = sc.eventID().run();                                                                             \
      auto lumi = sc.eventID().luminosityBlock();                                                                \
      stream_lumi_in_flight_ranges_.end(stream_domain_[sid], __func__, #signal, "sid run lumi", sid, run, lumi); \
    }                                                                                                            \
  }

// Macro for global (GlobalContext) signal pairs, keyed by the signal name via global_in_flight_ranges_.
#define DEFINE_GLOBAL_CONTEXT_SIGNAL_WATCHER(signal, color)                                            \
  template <class Backend>                                                                             \
  void ProfilerService<Backend>::pre##signal(edm::GlobalContext const&) {                              \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                \
      global_in_flight_ranges_.start(global_domain_, color, __func__, #signal, "", "signal", #signal); \
    }                                                                                                  \
  }                                                                                                    \
  template <class Backend>                                                                             \
  void ProfilerService<Backend>::post##signal(edm::GlobalContext const&) {                             \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                \
      global_in_flight_ranges_.end(global_domain_, __func__, #signal, "signal", #signal);              \
    }                                                                                                  \
  }

// Macro for global run (GlobalContext) signal pairs, keyed by run number.
#define DEFINE_GLOBAL_RUN_SIGNAL_WATCHER(signal, color)                                                 \
  template <class Backend>                                                                              \
  void ProfilerService<Backend>::pre##signal(edm::GlobalContext const& gc) {                            \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                 \
      auto run = gc.luminosityBlockID().run();                                                          \
      std::string detail = "runSlot=" + std::to_string(gc.runIndex().value());                          \
      global_run_in_flight_ranges_.start(global_domain_, color, __func__, #signal, detail, "run", run); \
    }                                                                                                   \
  }                                                                                                     \
  template <class Backend>                                                                              \
  void ProfilerService<Backend>::post##signal(edm::GlobalContext const& gc) {                           \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                 \
      auto run = gc.luminosityBlockID().run();                                                          \
      global_run_in_flight_ranges_.end(global_domain_, __func__, #signal, "run", run);                  \
    }                                                                                                   \
  }

// Macro for global lumi (GlobalContext) signal pairs, keyed by (run, lumi).
#define DEFINE_GLOBAL_LUMI_SIGNAL_WATCHER(signal, color)                                                            \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::GlobalContext const& gc) {                                        \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                             \
      auto run = gc.luminosityBlockID().run();                                                                      \
      auto lumi = gc.luminosityBlockID().luminosityBlock();                                                         \
      std::string detail = "runSlot=" + std::to_string(gc.runIndex().value()) +                                     \
                           " lumiSlot=" + std::to_string(gc.luminosityBlockIndex().value());                        \
      global_lumi_in_flight_ranges_.start(global_domain_, color, __func__, #signal, detail, "run lumi", run, lumi); \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::GlobalContext const& gc) {                                       \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                             \
      auto run = gc.luminosityBlockID().run();                                                                      \
      auto lumi = gc.luminosityBlockID().luminosityBlock();                                                         \
      global_lumi_in_flight_ranges_.end(global_domain_, __func__, #signal, "run lumi", run, lumi);                  \
    }                                                                                                               \
  }

// Macro for global no-argument signal pairs, keyed by the signal name. `guard` selects when to record.
#define DEFINE_GLOBAL_SIGNAL_WATCHER(signal, color, guard)                                             \
  template <class Backend>                                                                             \
  void ProfilerService<Backend>::pre##signal() {                                                       \
    if (guard) {                                                                                       \
      global_in_flight_ranges_.start(global_domain_, color, __func__, #signal, "", "signal", #signal); \
    }                                                                                                  \
  }                                                                                                    \
  template <class Backend>                                                                             \
  void ProfilerService<Backend>::post##signal() {                                                      \
    if (guard) {                                                                                       \
      global_in_flight_ranges_.end(global_domain_, __func__, #signal, "signal", #signal);              \
    }                                                                                                  \
  }

// Macro for global signal pairs taking (and ignoring) a single argument. `guard` selects when to record.
#define DEFINE_GLOBAL_ARG_SIGNAL_WATCHER(signal, color, guard, argType)                                \
  template <class Backend>                                                                             \
  void ProfilerService<Backend>::pre##signal(argType) {                                                \
    if (guard) {                                                                                       \
      global_in_flight_ranges_.start(global_domain_, color, __func__, #signal, "", "signal", #signal); \
    }                                                                                                  \
  }                                                                                                    \
  template <class Backend>                                                                             \
  void ProfilerService<Backend>::post##signal(argType) {                                               \
    if (guard) {                                                                                       \
      global_in_flight_ranges_.end(global_domain_, __func__, #signal, "signal", #signal);              \
    }                                                                                                  \
  }

// Macro for global signal pairs taking a file name (std::string), used as the range detail.
#define DEFINE_GLOBAL_STRING_SIGNAL_WATCHER(signal, color)                                              \
  template <class Backend>                                                                              \
  void ProfilerService<Backend>::pre##signal(std::string const& lfn) {                                  \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                 \
      global_in_flight_ranges_.start(global_domain_, color, __func__, #signal, lfn, "signal", #signal); \
    }                                                                                                   \
  }                                                                                                     \
  template <class Backend>                                                                              \
  void ProfilerService<Backend>::post##signal(std::string const&) {                                     \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                 \
      global_in_flight_ranges_.end(global_domain_, __func__, #signal, "signal", #signal);               \
    }                                                                                                   \
  }

// Macro for a single no-argument global mark (instantaneous annotation, not a range).
#define DEFINE_GLOBAL_MARK(name, color)          \
  template <class Backend>                       \
  void ProfilerService<Backend>::name() {        \
    Backend::mark(global_domain_, #name, color); \
  }

template <typename Backend>
void ProfilerService<Backend>::preallocate(edm::service::SystemBounds const& bounds) {
  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
      << bounds.maxNumberOfConcurrentLuminosityBlocks() << " luminosity sections, " << bounds.maxNumberOfStreams()
      << " streams\nrunning on " << bounds.maxNumberOfThreads() << " threads";
  Backend::mark(global_domain_, out.str().c_str(), Color::Grey);

  auto concurrentStreams = bounds.maxNumberOfStreams();
  // create the NVTX domains for per-EDM-stream transitions
  stream_domain_.resize(concurrentStreams);
  for (unsigned int sid = 0; sid < concurrentStreams; ++sid) {
    stream_domain_[sid].create(fmt::sprintf("EDM Stream %d", sid));
  }

  if (skipFirstEvent_) {
    globalFirstEventDone_ = false;
    std::vector<std::atomic<bool>> tmp(concurrentStreams);
    for (auto& element : tmp)
      std::atomic_init(&element, false);
    streamFirstEventDone_ = std::move(tmp);
  }
}

DEFINE_GLOBAL_MARK(postServicesConstruction, Color::Grey)

DEFINE_GLOBAL_SIGNAL_WATCHER(EventSetupConfigurationFinalized, Color::Blue, not skipFirstEvent_)

template <class Backend>
void ProfilerService<Backend>::eventSetupConfiguration(edm::eventsetup::ESRecordsToProductResolverIndices const&,
                                                       edm::ProcessContext const&) {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "eventSetupConfiguration", Color::Blue);
  }
}

DEFINE_GLOBAL_SIGNAL_WATCHER(EventSetupModulesConstruction, Color::Blue_Light2, not skipFirstEvent_)
DEFINE_GLOBAL_SIGNAL_WATCHER(ModulesAndSourceConstruction, Color::Green_Light2, not skipFirstEvent_)

/******** Job begin/end signal implementations *************************************/

template <typename Backend>
void ProfilerService<Backend>::preBeginJob(edm::ProcessContext const&) {
  global_in_flight_ranges_.start(global_domain_, Color::Grey, __func__, "BeginJob", "", "signal", "BeginJob");
}

template <typename Backend>
void ProfilerService<Backend>::postBeginJob() {
  global_in_flight_ranges_.end(global_domain_, __func__, "BeginJob", "signal", "BeginJob");
}

DEFINE_GLOBAL_SIGNAL_WATCHER(EndJob, Color::Grey, true)

template <class Backend>
void ProfilerService<Backend>::lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const&,
                                                            edm::ProcessContext const&) {
  Backend::mark(global_domain_, "lookupInitializationComplete", Color::Grey);
}

/******** Stream begin/end signal implementations *************************************/

DEFINE_STREAM_SIGNAL_WATCHER(BeginStream, Color::Grey)
DEFINE_STREAM_SIGNAL_WATCHER(EndStream, Color::Grey)

DEFINE_GLOBAL_MARK(jobFailure, Color::Red)

/******** Source transition signal implementations *************************************/

DEFINE_GLOBAL_SIGNAL_WATCHER(SourceNextTransition, Color::Yellow, true)

DEFINE_STREAM_ID_SIGNAL_WATCHER(SourceEvent, Color::Yellow)

DEFINE_GLOBAL_ARG_SIGNAL_WATCHER(SourceLumi,
                                 Color::Yellow,
                                 not skipFirstEvent_ or globalFirstEventDone_,
                                 edm::LuminosityBlockIndex)

DEFINE_GLOBAL_ARG_SIGNAL_WATCHER(SourceRun, Color::Yellow, not skipFirstEvent_ or globalFirstEventDone_, edm::RunIndex)

/******** Source process block signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preSourceProcessBlock() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    global_in_flight_ranges_.start(
        global_domain_, Color::Yellow, __func__, "SourceProcessBlock", "", "signal", "SourceProcessBlock");
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceProcessBlock(std::string const&) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    global_in_flight_ranges_.end(global_domain_, __func__, "SourceProcessBlock", "signal", "SourceProcessBlock");
  }
}

/******** File signal implementations *************************************/

DEFINE_GLOBAL_STRING_SIGNAL_WATCHER(OpenFile, Color::Amber)

DEFINE_GLOBAL_STRING_SIGNAL_WATCHER(CloseFile, Color::Amber)

DEFINE_GLOBAL_SIGNAL_WATCHER(OpenOutputFiles, Color::Amber, not skipFirstEvent_ or globalFirstEventDone_)

DEFINE_GLOBAL_SIGNAL_WATCHER(CloseOutputFiles, Color::Amber, not skipFirstEvent_ or globalFirstEventDone_)

/******** Global run/lumi and process block signal implementations *************************************/

DEFINE_GLOBAL_RUN_SIGNAL_WATCHER(GlobalBeginRun, Color::Grey)
DEFINE_GLOBAL_RUN_SIGNAL_WATCHER(GlobalEndRun, Color::Grey)
DEFINE_GLOBAL_LUMI_SIGNAL_WATCHER(GlobalBeginLumi, Color::Grey)
DEFINE_GLOBAL_LUMI_SIGNAL_WATCHER(GlobalEndLumi, Color::Grey)
DEFINE_GLOBAL_CONTEXT_SIGNAL_WATCHER(BeginProcessBlock, Color::Grey)
DEFINE_GLOBAL_CONTEXT_SIGNAL_WATCHER(EndProcessBlock, Color::Grey)
DEFINE_GLOBAL_CONTEXT_SIGNAL_WATCHER(AccessInputProcessBlock, Color::Grey)
DEFINE_GLOBAL_CONTEXT_SIGNAL_WATCHER(WriteProcessBlock, Color::Amber)
DEFINE_GLOBAL_RUN_SIGNAL_WATCHER(GlobalWriteRun, Color::Amber)
DEFINE_GLOBAL_LUMI_SIGNAL_WATCHER(GlobalWriteLumi, Color::Amber)

DEFINE_STREAM_RUN_SIGNAL_WATCHER(StreamBeginRun, Color::Grey)
DEFINE_STREAM_RUN_SIGNAL_WATCHER(StreamEndRun, Color::Grey)
DEFINE_STREAM_LUMI_SIGNAL_WATCHER(StreamBeginLumi, Color::Grey)
DEFINE_STREAM_LUMI_SIGNAL_WATCHER(StreamEndLumi, Color::Grey)

DEFINE_GLOBAL_MARK(beginProcessing, Color::Grey)

DEFINE_GLOBAL_MARK(endProcessing, Color::Grey)

/******** Event signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    std::string detail = fmt::sprintf(
        "run=%d lumi=%d event=%d", sc.eventID().run(), sc.eventID().luminosityBlock(), sc.eventID().event());
    event_in_flight_ranges_.start(stream_domain_[sid], Color::Yellow_Dark1, __func__, "Event", detail, "sid", sid);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_in_flight_ranges_.end(stream_domain_[sid], __func__, "Event", "sid", sid);
  } else {
    streamFirstEventDone_[sid] = true;
    auto identity = [](bool x) { return x; };
    if (std::all_of(streamFirstEventDone_.begin(), streamFirstEventDone_.end(), identity)) {
      bool expected = false;
      if (globalFirstEventDone_.compare_exchange_strong(expected, true)) {
        Backend::profilerStart();
        Backend::mark(global_domain_, "profiling started", Color::White);
      }
    }
  }
}

DEFINE_STREAM_SIGNAL_WATCHER(ClearEvent, Color::Yellow_Dark2)

template <class Backend>
void ProfilerService<Backend>::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    path_in_flight_ranges_.start(stream_domain_[sid],
                                 Color::Grey,
                                 __func__,
                                 "PathEvent",
                                 pc.pathName(),
                                 "sid pid isEndPath",
                                 sid,
                                 pid,
                                 pc.isEndPath());
  }
}

template <class Backend>
void ProfilerService<Backend>::postPathEvent(edm::StreamContext const& sc,
                                             edm::PathContext const& pc,
                                             edm::HLTPathStatus const&) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    path_in_flight_ranges_.end(
        stream_domain_[sid], __func__, "PathEvent", "sid pid isEndPath", sid, pid, pc.isEndPath());
  }
}

/******** Module construction / job signal implementations *************************************/

DEFINE_MODULE_DESC_SIGNAL_WATCHER(ModuleConstruction, Color::Green_Light2, not skipFirstEvent_)
DEFINE_MODULE_DESC_SIGNAL_WATCHER(ModuleDestruction, Color::Green_Light2, not skipFirstEvent_)
DEFINE_MODULE_DESC_SIGNAL_WATCHER(ModuleBeginJob, Color::Green_Dark1, not skipFirstEvent_)
DEFINE_MODULE_DESC_SIGNAL_WATCHER(SourceConstruction, Color::Yellow_Light2, not skipFirstEvent_)

DEFINE_MODULE_DESC_SIGNAL_WATCHER(ModuleEndJob, Color::Green_Dark1, not skipFirstEvent_ or globalFirstEventDone_)

/******** Module stream context signal implementations *********************************************/

DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventPrefetching, stream_modules_in_flight_ranges_, Color::Green_Light1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent, stream_modules_event_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire, stream_modules_event_acquire_in_flight_ranges_, Color::Green)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransformPrefetching, Color::Green)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransform, Color::Green_Dark2)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransformAcquiring, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamPrefetching, stream_modules_in_flight_ranges_, Color::Green_Light1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi, stream_modules_in_flight_ranges_, Color::Green_Dark1)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi, stream_modules_in_flight_ranges_, Color::Green_Dark1)

/******** Module global run/lumi and process block signal implementations *********************************/

DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalBeginRun, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalEndRun, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalBeginLumi, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalEndLumi, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleBeginProcessBlock, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleAccessInputProcessBlock, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleEndProcessBlock, Color::Green_Dark1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalPrefetching, Color::Green_Light1)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteProcessBlock, Color::Amber)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteRun, Color::Amber)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteLumi, Color::Amber)

/******** ES module signal implementations *************************************/

DEFINE_ES_CONSTRUCTION_SIGNAL_WATCHER(ESModuleConstruction, Color::Blue_Light2)

template <class Backend>
void ProfilerService<Backend>::postESModuleRegistration(
    edm::eventsetup::ComponentDescription const& componentDescription) {
  auto const& label = componentDescription.label_;
  auto const& msg = label + " " + "ESModuleReRegistration";
  Backend::mark(global_domain_, msg.c_str(), Color::Blue_Light2);
}

template <class Backend>
void ProfilerService<Backend>::preESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                      edm::ESModuleCallingContext const& esmcc) {
  preESModuleAcquire(iKey, esmcc);
}

template <class Backend>
void ProfilerService<Backend>::postESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                       edm::ESModuleCallingContext const& esmcc) {
  postESModuleAcquire(iKey, esmcc);
}

DEFINE_ES_SIGNAL_WATCHER(ESModule, Color::Blue_Dark1)
DEFINE_ES_SIGNAL_WATCHER(ESModuleAcquire, Color::Blue)

/******** ES IOV sync signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::esSyncIOVQueuing(edm::IOVSyncValue const&) {
  Backend::mark(global_domain_, "esSyncIOVQueuing", Color::Blue);
}

DEFINE_GLOBAL_ARG_SIGNAL_WATCHER(ESSyncIOV, Color::Blue, true, edm::IOVSyncValue const&)

/******** Infrastructure/setup signal implementations *************************************/

DEFINE_GLOBAL_SIGNAL_WATCHER(FinishSchedule, Color::Grey, not skipFirstEvent_)
DEFINE_GLOBAL_SIGNAL_WATCHER(PrincipalsCreation, Color::Grey, not skipFirstEvent_)
DEFINE_GLOBAL_SIGNAL_WATCHER(ScheduleConsistencyCheck, Color::Grey, not skipFirstEvent_)
DEFINE_GLOBAL_SIGNAL_WATCHER(ModulesInitializationFinalized, Color::Grey, not skipFirstEvent_)

/******** Early termination signal implementations *****************************/

template <class Backend>
void ProfilerService<Backend>::preStreamEarlyTermination(edm::StreamContext const& sc, edm::TerminationOrigin) {
  auto sid = sc.streamID();
  Backend::mark(stream_domain_[sid], "early termination", Color::Red);
}

template <class Backend>
void ProfilerService<Backend>::preGlobalEarlyTermination(edm::GlobalContext const&, edm::TerminationOrigin) {
  Backend::mark(global_domain_, "global early termination", Color::Red);
}

template <class Backend>
void ProfilerService<Backend>::preSourceEarlyTermination(edm::TerminationOrigin) {
  Backend::mark(global_domain_, "source early termination", Color::Red);
}

#undef DEFINE_ES_SIGNAL_WATCHER
#undef DEFINE_MODULE_STREAM_SIGNAL_WATCHER
#undef DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER
#undef DEFINE_MODULE_DESC_SIGNAL_WATCHER
#undef DEFINE_STREAM_SIGNAL_WATCHER
#undef DEFINE_STREAM_RUN_SIGNAL_WATCHER
#undef DEFINE_STREAM_LUMI_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_CONTEXT_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_RUN_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_LUMI_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_ARG_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_STRING_SIGNAL_WATCHER
#undef DEFINE_ES_CONSTRUCTION_SIGNAL_WATCHER
#undef DEFINE_STREAM_ID_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_MARK

#endif  // __FWCore_Services_ProfilerService_h__

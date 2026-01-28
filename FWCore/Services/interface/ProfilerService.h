#ifndef __FWCore_Services_ProfilerService_h__
#define __FWCore_Services_ProfilerService_h__

/**
 * Based template class for range/mark based profiling services, targeting
 * NVidia NVTX, AMP ROCmTX, or VTune.
 */

/**
 * Helper marcos to declare similar functions (for pre/post couples).
 */

#define DECLARE_ES_SIGNAL_WATCHER(signal)                                                                     \
  void pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc); \
  void post##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc);

// The global_es_modules_ vector is indexed by the ComponentDescription id_ field
#define DEFINE_ES_SIGNAL_WATCHER(signal)                                                        \
  template <class Backend>                                                                      \
  void ProfilerService<Backend>::pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey,  \
                                             edm::ESModuleCallingContext const& esmcc) {        \
    auto mid = esmcc.componentDescription()->id_;                                               \
    auto const& label = esmcc.componentDescription()->label_;                                   \
    auto const& type = esmcc.componentDescription()->type_;                                     \
    std::string msg;                                                                            \
    if (label.empty()) {                                                                        \
      /*Fallback on the type */                                                                 \
      msg = type + "(type) " + #signal "";                                                      \
    } else {                                                                                    \
      msg = label + " " + #signal "";                                                           \
    }                                                                                           \
    global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), Color::Blue, __func__);   \
  }                                                                                             \
  template <class Backend>                                                                      \
  void ProfilerService<Backend>::post##signal(edm::eventsetup::EventSetupRecordKey const& iKey, \
                                              edm::ESModuleCallingContext const& esmcc) {       \
    auto mid = esmcc.componentDescription()->id_;                                               \
    auto const& label = esmcc.componentDescription()->label_;                                   \
    auto const& type = esmcc.componentDescription()->type_;                                     \
    std::string msg;                                                                            \
    if (label.empty()) {                                                                        \
      /* Fallback on the type */                                                                \
      msg = type + "(type) " + #signal "";                                                      \
    } else {                                                                                    \
      msg = label + " " + #signal "";                                                           \
    }                                                                                           \
    global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);                       \
  }

#define DECLARE_MODULE_STREAM_SIGNAL_WATCHER(signal)                                    \
  void pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc); \
  void post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc);

#define DEFINE_MODULE_STREAM_SIGNAL_WATCHER(signal)                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {  \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const& msg = label + " " + #signal "";                                                                   \
      stream_modules_[sid][mid].startColorIn(stream_domain_[sid], msg.c_str(), labelColor(label), __func__);        \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const& msg = label + " " + #signal "";                                                                   \
      stream_modules_[sid][mid].endIn(stream_domain_[sid], msg.c_str(), __func__);                                  \
    }                                                                                                               \
  }

// This macro registers signal watchers pairs. Same for all.
#define REGISTER_SIGNAL_WATCHER(signal)                           \
  registry.watchPre##signal(this, &ProfilerService::pre##signal); \
  registry.watchPost##signal(this, &ProfilerService::post##signal);

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

using namespace std::string_literals;

/**
   * @brief Abstract color enumeration the derived classes can translate (or disregard).
   */
enum class ProfilerServiceColor : std::size_t {
  Black = 0,
  Red,
  DarkGreen,
  Green,
  LightGreen,
  Blue,
  Amber,
  LightAmber,
  White
};

[[maybe_unused]] static size_t to_underlying(ProfilerServiceColor c) noexcept { return static_cast<std::size_t>(c); }

template <typename Backend>
class ProfilerService {
public:
  using Color = ProfilerServiceColor;
  using Range = typename Backend::Range;
  using Domain = typename Backend::Domain;

  ProfilerService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~ProfilerService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void preallocate(edm::service::SystemBounds const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preBeginJob(edm::ProcessContext const&);
  void postBeginJob();

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

  void preEndJob();
  void postEndJob();

  /******* Global context signals  **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalBeginRun(edm::GlobalContext const&);
  void postGlobalBeginRun(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalEndRun(edm::GlobalContext const&);
  void postGlobalEndRun(edm::GlobalContext const&);

  /******* Stream context signals  **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamBeginRun(edm::StreamContext const&);
  void postStreamBeginRun(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamEndRun(edm::StreamContext const&);
  void postStreamEndRun(edm::StreamContext const&);

  /******** Global context lumi signals **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalBeginLumi(edm::GlobalContext const&);
  void postGlobalBeginLumi(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalEndLumi(edm::GlobalContext const&);
  void postGlobalEndLumi(edm::GlobalContext const&);

  /******** Stream context lumi signals **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamBeginLumi(edm::StreamContext const&);
  void postStreamBeginLumi(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamEndLumi(edm::StreamContext const&);
  void postStreamEndLumi(edm::StreamContext const&);

  /******** Stream context events signal **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preEvent(edm::StreamContext const&);
  void postEvent(edm::StreamContext const&);

  void preClearEvent(edm::StreamContext const&);
  void postClearEvent(edm::StreamContext const&);

  /******** Path context event signals **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void prePathEvent(edm::StreamContext const&, edm::PathContext const&);
  void postPathEvent(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

  /******** Module context signals *********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);

  /******** File context signals **********************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preOpenFile(std::string const&);
  void postOpenFile(std::string const&);

  // these signal pair are guaranteed to be called by the same thread
  void preCloseFile(std::string const&);
  void postCloseFile(std::string const&);

  /******** Source transition signals *********************************************/

  void preSourceNextTransition();
  void postSourceNextTransition();

  /******** Source module context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceConstruction(edm::ModuleDescription const&);
  void postSourceConstruction(edm::ModuleDescription const&);

  /******** Source run context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceRun(edm::RunIndex);
  void postSourceRun(edm::RunIndex);

  /******** Source lumi context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceLumi(edm::LuminosityBlockIndex);
  void postSourceLumi(edm::LuminosityBlockIndex);

  /******** Source stream context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceEvent(edm::StreamID);
  void postSourceEvent(edm::StreamID);

  /******** Module no-context signals *********************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preModuleConstruction(edm::ModuleDescription const&);
  void postModuleConstruction(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleDestruction(edm::ModuleDescription const&);
  void postModuleDestruction(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleBeginJob(edm::ModuleDescription const&);
  void postModuleBeginJob(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEndJob(edm::ModuleDescription const&);
  void postModuleEndJob(edm::ModuleDescription const&);

  /******** Module global context signals *********************************************/

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

  /******** Module stream context signals *********************************************/

  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformPrefetching)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformAcquiring)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransform)

  /******** ES module context signals *********************************************/
  // ES signal watchers
  void postESModuleRegistration(edm::eventsetup::ComponentDescription const&);
  // Prefetching is optionally watched
  // (see constructor)
  DECLARE_ES_SIGNAL_WATCHER(ESModulePrefetching)
  DECLARE_ES_SIGNAL_WATCHER(ESModule)
  DECLARE_ES_SIGNAL_WATCHER(ESModuleAcquire)

private:
  bool highlight(std::string const& label) const {
    return (std::binary_search(highlightModules_.begin(), highlightModules_.end(), label));
  }

  Color labelColor(std::string const& label) const { return highlight(label) ? Color::Amber : Color::Green; }

  Color labelColorLight(std::string const& label) const {
    return highlight(label) ? Color::LightAmber : Color::LightGreen;
  }

  std::vector<std::string> highlightModules_;
  const bool showModulePrefetching_;
  const bool skipFirstEvent_;

  std::atomic<bool> globalFirstEventDone_ = false;
  std::vector<std::atomic<bool>> streamFirstEventDone_;
  Range globalRange_;          // global event range
  std::vector<Range> event_;   // per-stream event ranges
  std::vector<Range> source_;  // per-stream source ranges TODO: it might be possible to merge this with event_
  std::vector<std::vector<Range>> path_;            // per-stream, per-path ranges
  std::vector<std::vector<Range>> endPath_;         // per-stream, per-endPath ranges
  std::vector<std::vector<Range>> stream_modules_;  // per-stream, per-module ranges
  std::vector<std::vector<Range>>
      stream_modules_acquire_;  // per-stream, per-module ranges for acquire, which can clash with produce
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<Range> global_modules_;       // global per-module events
  std::vector<std::vector<Range>> stream_ES_modules_;  // per-stream, per-ES-module ranges
  std::vector<std::vector<Range>>
      stream_ES_modules_acquire_;  // per-stream, per-ES-module ranges for acquire, which can clash with produce
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<Range> global_ES_modules_;  // global per-ES-module events

  Domain global_domain_;               // NVTX domain for global EDM transitions
  std::vector<Domain> stream_domain_;  // NVTX domains for per-EDM-stream transitions
};

template <typename Backend>
ProfilerService<Backend>::ProfilerService(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
    : highlightModules_(config.getUntrackedParameter<std::vector<std::string>>("highlightModules")),
      showModulePrefetching_(config.getUntrackedParameter<bool>("showModulePrefetching")),
      skipFirstEvent_(config.getUntrackedParameter<bool>("skipFirstEvent")) {
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

  registry.watchPreallocate(this, &ProfilerService::preallocate);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreBeginJob(this, &ProfilerService::preBeginJob);
  registry.watchPostBeginJob(this, &ProfilerService::postBeginJob);

  registry.watchLookupInitializationComplete(this, &ProfilerService::lookupInitializationComplete);

  registry.watchPreEndJob(this, &ProfilerService::preEndJob);
  registry.watchPostEndJob(this, &ProfilerService::postEndJob);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalBeginRun(this, &ProfilerService::preGlobalBeginRun);
  registry.watchPostGlobalBeginRun(this, &ProfilerService::postGlobalBeginRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalEndRun(this, &ProfilerService::preGlobalEndRun);
  registry.watchPostGlobalEndRun(this, &ProfilerService::postGlobalEndRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamBeginRun(this, &ProfilerService::preStreamBeginRun);
  registry.watchPostStreamBeginRun(this, &ProfilerService::postStreamBeginRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamEndRun(this, &ProfilerService::preStreamEndRun);
  registry.watchPostStreamEndRun(this, &ProfilerService::postStreamEndRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalBeginLumi(this, &ProfilerService::preGlobalBeginLumi);
  registry.watchPostGlobalBeginLumi(this, &ProfilerService::postGlobalBeginLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalEndLumi(this, &ProfilerService::preGlobalEndLumi);
  registry.watchPostGlobalEndLumi(this, &ProfilerService::postGlobalEndLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamBeginLumi(this, &ProfilerService::preStreamBeginLumi);
  registry.watchPostStreamBeginLumi(this, &ProfilerService::postStreamBeginLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamEndLumi(this, &ProfilerService::preStreamEndLumi);
  registry.watchPostStreamEndLumi(this, &ProfilerService::postStreamEndLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreEvent(this, &ProfilerService::preEvent);
  registry.watchPostEvent(this, &ProfilerService::postEvent);

  registry.watchPreClearEvent(this, &ProfilerService::preClearEvent);
  registry.watchPostClearEvent(this, &ProfilerService::postClearEvent);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPrePathEvent(this, &ProfilerService::prePathEvent);
  registry.watchPostPathEvent(this, &ProfilerService::postPathEvent);

  if (showModulePrefetching_) {
    // these signal pair are NOT guaranteed to be called by the same thread
    registry.watchPreModuleEventPrefetching(this, &ProfilerService::preModuleEventPrefetching);
    registry.watchPostModuleEventPrefetching(this, &ProfilerService::postModuleEventPrefetching);
  }

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreOpenFile(this, &ProfilerService::preOpenFile);
  registry.watchPostOpenFile(this, &ProfilerService::postOpenFile);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreCloseFile(this, &ProfilerService::preCloseFile);
  registry.watchPostCloseFile(this, &ProfilerService::postCloseFile);

  registry.watchPreSourceNextTransition(this, &ProfilerService::preSourceNextTransition);
  registry.watchPostSourceNextTransition(this, &ProfilerService::postSourceNextTransition);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceConstruction(this, &ProfilerService::preSourceConstruction);
  registry.watchPostSourceConstruction(this, &ProfilerService::postSourceConstruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceRun(this, &ProfilerService::preSourceRun);
  registry.watchPostSourceRun(this, &ProfilerService::postSourceRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceLumi(this, &ProfilerService::preSourceLumi);
  registry.watchPostSourceLumi(this, &ProfilerService::postSourceLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceEvent(this, &ProfilerService::preSourceEvent);
  registry.watchPostSourceEvent(this, &ProfilerService::postSourceEvent);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleConstruction(this, &ProfilerService::preModuleConstruction);
  registry.watchPostModuleConstruction(this, &ProfilerService::postModuleConstruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleDestruction(this, &ProfilerService::preModuleDestruction);
  registry.watchPostModuleDestruction(this, &ProfilerService::postModuleDestruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalBeginRun(this, &ProfilerService::preModuleGlobalBeginRun);
  registry.watchPostModuleGlobalBeginRun(this, &ProfilerService::postModuleGlobalBeginRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalEndRun(this, &ProfilerService::preModuleGlobalEndRun);
  registry.watchPostModuleGlobalEndRun(this, &ProfilerService::postModuleGlobalEndRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalBeginLumi(this, &ProfilerService::preModuleGlobalBeginLumi);
  registry.watchPostModuleGlobalBeginLumi(this, &ProfilerService::postModuleGlobalBeginLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalEndLumi(this, &ProfilerService::preModuleGlobalEndLumi);
  registry.watchPostModuleGlobalEndLumi(this, &ProfilerService::postModuleGlobalEndLumi);

  /******** Module stream context signals *********************************************/

  REGISTER_SIGNAL_WATCHER(ModuleBeginJob)
  REGISTER_SIGNAL_WATCHER(ModuleEndJob)
  REGISTER_SIGNAL_WATCHER(ModuleBeginStream)
  REGISTER_SIGNAL_WATCHER(ModuleEndStream)
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndLumi)
  REGISTER_SIGNAL_WATCHER(ModuleEventAcquire)
  REGISTER_SIGNAL_WATCHER(ModuleEvent)
  REGISTER_SIGNAL_WATCHER(ModuleEventDelayedGet)
  REGISTER_SIGNAL_WATCHER(EventReadFromSource)
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleTransformPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleTransformAcquiring)
  REGISTER_SIGNAL_WATCHER(ModuleTransform)

  // ES signal watchers
  registry.watchPostESModuleRegistration(this, &ProfilerService::postESModuleRegistration);
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ESModulePrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ESModule)
  REGISTER_SIGNAL_WATCHER(ESModuleAcquire)
}

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
  descriptions.add(Backend::shortName() + "ProfilerService", desc);
  descriptions.setComment(Backend::serviceComment());
  // For reference, here is a possible extended comment for nvprof/nvvm backends:
  //   descriptions.setComment(R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

  // Notes on nvprof options:
  //   - the option '--profile-from-start off' should be used if skipFirstEvent is True.
  //   - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
  //   - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)");
}

template <typename Backend>
void ProfilerService<Backend>::preallocate(edm::service::SystemBounds const& bounds) {
  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
      << bounds.maxNumberOfConcurrentLuminosityBlocks() << " luminosity sections, " << bounds.maxNumberOfStreams()
      << " streams\nrunning on " << bounds.maxNumberOfThreads() << " threads";
  Backend::mark(global_domain_, out.str().c_str(), Color::Amber);

  auto concurrentStreams = bounds.maxNumberOfStreams();
  // create the NVTX domains for per-EDM-stream transitions
  stream_domain_.resize(concurrentStreams);
  for (unsigned int sid = 0; sid < concurrentStreams; ++sid) {
    stream_domain_[sid].create(fmt::sprintf("EDM Stream %d", sid));
  }

  event_.resize(concurrentStreams);
  path_.resize(concurrentStreams);
  endPath_.resize(concurrentStreams);
  source_.resize(concurrentStreams);
  // per stream path and end path arrays will be resized in lookupInitializationComplete()
  stream_modules_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_) {
    modulesForOneStream.resize(global_modules_.size());
  }
  stream_modules_acquire_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_acquire_) {
    modulesForOneStream.resize(global_modules_.size());
  }

  if (skipFirstEvent_) {
    globalFirstEventDone_ = false;
    std::vector<std::atomic<bool>> tmp(concurrentStreams);
    for (auto& element : tmp)
      std::atomic_init(&element, false);
    streamFirstEventDone_ = std::move(tmp);
  }
}

template <typename Backend>
void ProfilerService<Backend>::preBeginJob(edm::ProcessContext const& context) {
  globalRange_.startColorIn(global_domain_, "preBeginJob", Color::Amber, __func__);
}

template <typename Backend>
void ProfilerService<Backend>::postBeginJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "postBeginJob", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                                            edm::ProcessContext const&) {
  Backend::mark(global_domain_, "lookupInitializationComplete", Color::Amber);
  // We could potentially get all we want from pathsAndConsumes...
  assert(!path_.empty() and !endPath_.empty());
  for (auto& streamPaths : path_) {
    streamPaths.resize(pathsAndConsumes.paths().size());
  }
  for (auto& streamEndPaths : endPath_) {
    streamEndPaths.resize(pathsAndConsumes.endPaths().size());
  }
}

template <class Backend>
void ProfilerService<Backend>::preEndJob() {
  globalRange_.startColorIn(global_domain_, "EndJob", Color::Amber, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postEndJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "EndJob", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    source_[sid].startColorIn(stream_domain_[sid], "source", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    source_[sid].endIn(stream_domain_[sid], "source", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preOpenFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, ("open file "s + lfn).c_str(), Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postOpenFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, ("open file "s + lfn).c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preCloseFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, ("close file "s + lfn).c_str(), Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postCloseFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, ("close file "s + lfn).c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global begin run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global begin run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global end run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global end run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream begin run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream begin run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream end run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream end run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global begin lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global begin lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global end lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global end lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream begin lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream begin lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream end lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream end lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    std::string msg = fmt::sprintf("event run = %d event = %d", sc.eventID().run(), sc.eventID().event());
    event_[sid].startColorIn(stream_domain_[sid], "event", Color::DarkGreen, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "event", __func__);
  } else {
    streamFirstEventDone_[sid] = true;
    auto identity = [](bool x) { return x; };
    if (std::all_of(streamFirstEventDone_.begin(), streamFirstEventDone_.end(), identity)) {
      bool expected = false;
      if (globalFirstEventDone_.compare_exchange_strong(expected, true))
        Backend::profilerStart();
    }
  }
}

template <class Backend>
void ProfilerService<Backend>::preClearEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "clear event", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postClearEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "clear event", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  auto& pathOrEndPath = pc.isEndPath() ? endPath_[sid][pid] : path_[sid][pid];
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    pathOrEndPath.startColorIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), Color::DarkGreen, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postPathEvent(edm::StreamContext const& sc,
                                             edm::PathContext const& pc,
                                             edm::HLTPathStatus const& hlts) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  auto& pathOrEndPath = pc.isEndPath() ? endPath_[sid][pid] : path_[sid][pid];
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    pathOrEndPath.endIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleConstruction(edm::ModuleDescription const& desc) {
  auto mid = desc.id();
  global_modules_.grow_to_at_least(mid + 1);
  std::cout << "ProfilerService::preModuleConstruction: module id " << mid << ", label: " << desc.moduleLabel() << "\n";

  // This normally does nothing because stream_modules_ is empty when
  // called. But there is a rare case when a looper is used that replacement
  // modules can be constructed at end of loop. I'm not sure if that feature
  // is ever actually used but just to be safe...
  for (auto& modulesForOneStream : stream_modules_) {
    modulesForOneStream.resize(global_modules_.size());
  }

  if (not skipFirstEvent_) {
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleDestruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " destruction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleDestruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " destruction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " begin job";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " begin job";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " end job";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " end job";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

/******** Module stream context signals *********************************************/

DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventPrefetching)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformPrefetching)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformAcquiring)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransform)

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalBeginRun(edm::GlobalContext const& gc,
                                                       edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin run";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalBeginRun(edm::GlobalContext const& gc,
                                                        edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin run";
    global_modules_[mid].endIn(global_domain_, "", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalEndRun(edm::GlobalContext const& gc,
                                                     edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end run";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalEndRun(edm::GlobalContext const& gc,
                                                      edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end run";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalBeginLumi(edm::GlobalContext const& gc,
                                                        edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin lumi";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalBeginLumi(edm::GlobalContext const& gc,
                                                         edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin lumi";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalEndLumi(edm::GlobalContext const& gc,
                                                      edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end lumi";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalEndLumi(edm::GlobalContext const& gc,
                                                       edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end lumi";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preSourceNextTransition() {
  globalRange_.startColorIn(global_domain_, "source transition", Color::Amber, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postSourceNextTransition() {
  globalRange_.endIn(global_domain_, "source transition", __func__);
}

template <class Backend>
void ProfilerService<Backend>::preSourceConstruction(edm::ModuleDescription const& desc) {
  auto mid = desc.id();
  global_modules_.grow_to_at_least(mid + 1);

  if (not skipFirstEvent_) {
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postESModuleRegistration(
    edm::eventsetup::ComponentDescription const& componentDescription) {
  auto mid = componentDescription.id_;
  auto const& label = componentDescription.label_;
  auto const& msg = label + " " + "ESModuleReRegistration";
  global_ES_modules_.grow_to_at_least(mid + 1);
  Backend::mark(global_domain_, msg.c_str(), Color::Amber);
}

template <class Backend>
void ProfilerService<Backend>::preESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                      edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  std::string msg;
  if (label.empty()) {
    // Fallback on the type
    msg = type + "(type) " +
          "ES prefetch"
          " acquire";
  } else {
    msg = label + " " +
          "ES prefetch"
          " acquire";
  }
  global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), Color::Blue, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                       edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  std::string msg;
  if (label.empty()) {
    // Fallback on the type
    msg = type + "(type) " +
          "ES prefetch"
          " acquire";
  } else {
    msg = label + " " +
          "ES prefetch"
          " acquire";
  }
  global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
}

/*DEFINE_ES_SIGNAL_WATCHER(ESModule)*/
template <class Backend>
void ProfilerService<Backend>::preESModule(edm::eventsetup::EventSetupRecordKey const& iKey,
                                           edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  auto const& context = iKey.name();
  std::string msg = "ESModule: label = '" + label + "', type = '" + type + "', record = '" + context + "'";
  global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), Color::Blue, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postESModule(edm::eventsetup::EventSetupRecordKey const& iKey,
                                            edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  auto const& context = iKey.name();
  std::string msg = "ESModule: label = '" + label + "', type = '" + type + "', record = '" + context + "'";
  global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
}

DEFINE_ES_SIGNAL_WATCHER(ESModuleAcquire)

#undef DECLARE_ES_SIGNAL_WATCHER
#undef DEFINE_ES_SIGNAL_WATCHER
#undef DECLARE_MODULE_STREAM_SIGNAL_WATCHER
#undef DEFINE_MODULE_STREAM_SIGNAL_WATCHER
#undef REGISTER_SIGNAL_WATCHER

#endif  // __FWCore_Services_ProfilerService_h__

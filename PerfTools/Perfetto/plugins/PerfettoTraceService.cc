// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
#include "PerfTools/Perfetto/interface/PerfettoAllocatorMonitor.h"
#include <perfetto.h>
#include "PerfTools/Perfetto/plugins/PerfettoCuptiProfiler.h"
#include "PerfTools/Perfetto/plugins/PerfettoPowerSampler.h"

#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocatorMonitor.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include <algorithm>
#include <atomic>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

// PerfettoTraceService records a Perfetto trace (.pftrace) of a cmsRun job using
// the in-process Perfetto SDK.
//
// Threading model (this is the important part). CMSSW runs one global TBB arena;
// within a *single* stream and a *single* event, independent modules execute as
// concurrent tasks on different threads, and an ExternalWork module's acquire()
// and produce() run on different threads with an asynchronous gap in between. A
// single per-stream timeline therefore cannot represent module execution without
// overlapping (mis-paired) slices. So:
//
//   * module / acquire / EventSetup / source / cleanup slices are emitted on a
//     per-(stream, thread) lane that hangs under the stream track (see
//     CMSSWPerfettoLanes.h). Each lane is fed by exactly one thread, so its slices
//     nest correctly; concurrent work in a stream lands on separate lanes.
//   * the per-stream "Event" lifetime (preEvent..postClearEvent) IS serialized
//     per stream, so it sits on the stream track itself, together with the
//     run/lumi/event counters.
//
// Around every module call the service also publishes a thread-local
// cms::perfetto::ModuleContext, so the caching-allocator monitor and the CUPTI
// GPU layer can attribute their work to the responsible module.

class PerfettoTraceService {
public:
  PerfettoTraceService(edm::ParameterSet const& pset, edm::ActivityRegistry& ar)
      : enabled_(pset.getUntrackedParameter<bool>("enabled")),
        fileName_(pset.getUntrackedParameter<std::string>("fileName")),
        bufferSizeKB_(pset.getUntrackedParameter<unsigned>("bufferSizeKB")),
        shmemSizeKB_(pset.getUntrackedParameter<unsigned>("shmemSizeKB")),
        maxEvents_(pset.getUntrackedParameter<unsigned>("maxEvents")),
        traceFunctions_(pset.getUntrackedParameter<bool>("traceFunctions")),
        traceAllocations_(pset.getUntrackedParameter<bool>("traceAllocations")),
        traceGpuKernels_(pset.getUntrackedParameter<bool>("traceGpuKernels")),
        tracePower_(pset.getUntrackedParameter<bool>("tracePower")),
        powerPeriodMs_(pset.getUntrackedParameter<unsigned>("powerPeriodMs")),
        traceModules_(pset.getUntrackedParameter<std::vector<std::string>>("traceModules")) {
    std::sort(traceModules_.begin(), traceModules_.end());
    if (!enabled_)
      return;

    ::perfetto::TracingInitArgs args;
    args.backends = ::perfetto::kInProcessBackend;
    // Size the producer shared-memory buffer (SMB) for the high, bursty slice rate
    // of many concurrent edm::streams. With the SDK default (~4 MB) the SMB
    // saturates and TrackEvent (kDrop policy) silently discards slices -- dropping
    // whole events from the trace. 32 KB chunks (the SDK maximum) reduce per-chunk
    // contention under heavy multithreading. kDrop is kept (not kStall) so tracing
    // never blocks the reconstruction threads.
    args.shmem_size_hint_kb = shmemSizeKB_;
    args.shmem_page_size_hint_kb = 32;
    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();

    ::perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(bufferSizeKB_);

    auto* ds = cfg.add_data_sources();
    auto* ds_cfg = ds->mutable_config();
    ds_cfg->set_name("track_event");

    ::perfetto::protos::gen::TrackEventConfig te;
    te.add_enabled_categories("cmssw.event");
    te.add_enabled_categories("cmssw.source");
    te.add_enabled_categories("cmssw.module");
    te.add_enabled_categories("cmssw.acquire");
    te.add_enabled_categories("cmssw.es");
    te.add_enabled_categories("cmssw.cleanup");
    te.add_enabled_categories("cmssw.func");
    te.add_enabled_categories("cmssw.alloc");
    te.add_enabled_categories("cmssw.gpu");
    te.add_enabled_categories("cmssw.power");
    ds_cfg->set_track_event_config_raw(te.SerializeAsString());

    // In-memory session: the whole trace is held in the buffer and written once at
    // the end. This keeps the GPU/CUPTI events -- emitted at end-of-job with their
    // real, earlier device timestamps -- correctly ordered, and avoids perfetto's
    // write_into_file "no flush" warning. The trace is bounded by bufferSizeKB;
    // raise it for very long jobs.
    session_ = ::perfetto::Tracing::NewTrace();
    session_->Setup(cfg);
    session_->StartBlocking();

    {
      auto proc = ::perfetto::ProcessTrack::Current();
      auto desc = proc.Serialize();
      desc.mutable_process()->set_process_name("cmsRun");
      ::perfetto::TrackEvent::SetTrackDescriptor(proc, desc);
    }

    ar.watchPreallocate(this, &PerfettoTraceService::preallocate);

    ar.watchPreSourceEvent(this, &PerfettoTraceService::preSourceEvent);
    ar.watchPostSourceEvent(this, &PerfettoTraceService::postSourceEvent);

    ar.watchPreEvent(this, &PerfettoTraceService::preEvent);
    ar.watchPreClearEvent(this, &PerfettoTraceService::preClearEvent);
    ar.watchPostClearEvent(this, &PerfettoTraceService::postClearEvent);

    ar.watchPreModuleEvent(this, &PerfettoTraceService::preModuleEvent);
    ar.watchPostModuleEvent(this, &PerfettoTraceService::postModuleEvent);
    ar.watchPreModuleEventAcquire(this, &PerfettoTraceService::preModuleEventAcquire);
    ar.watchPostModuleEventAcquire(this, &PerfettoTraceService::postModuleEventAcquire);

    ar.watchPreESModule(this, &PerfettoTraceService::preESModule);
    ar.watchPostESModule(this, &PerfettoTraceService::postESModule);

    ar.watchPostEndJob(this, &PerfettoTraceService::postEndJob);

    // Observe caching-allocator transactions (device/host memory). Registered
    // last so it is active for the whole job.
    if (traceAllocations_)
      cms::alpakatools::setCachingAllocatorMonitor(&allocatorMonitor_);

    // Stream CUDA kernel activity (real device-side timing + register/occupancy
    // info) into the session via CUPTI; a no-op when no GPU is present.
    if (traceGpuKernels_)
      cuptiProfiler_.start();

    // Sample CPU (RAPL) and GPU (NVML) power on a background thread; a no-op when
    // neither source is available.
    if (tracePower_)
      powerSampler_.start(powerPeriodMs_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("enabled", true)->setComment("Master switch; when false the service does nothing.");
    desc.addUntracked<std::string>("fileName", "cmsrun.pftrace")->setComment("Output Perfetto trace file.");
    desc.addUntracked<unsigned>("bufferSizeKB", 256 * 1024)->setComment("In-process trace buffer size in KB.");
    desc.addUntracked<unsigned>("shmemSizeKB", 64 * 1024)
        ->setComment(
            "Producer shared-memory buffer (SMB) size in KB; if too small the trace silently drops "
            "slices (whole events) under high-rate multi-stream tracing.");
    desc.addUntracked<unsigned>("maxEvents", 0)
        ->setComment(
            "Stop opening new event slices after this many events (0 = unlimited, the default; the "
            "trace is then bounded by bufferSizeKB).");
    desc.addUntracked<bool>("traceFunctions", false)
        ->setComment("Enable tier-B per-function slices (CMS_PERFETTO_FUNC/SCOPE).");
    desc.addUntracked<bool>("traceAllocations", false)
        ->setComment("Trace Alpaka caching-allocator alloc/free and device-memory counters.");
    desc.addUntracked<bool>("traceGpuKernels", false)
        ->setComment("Trace CUDA kernels (real device timing + registers/occupancy) via CUPTI.");
    desc.addUntracked<bool>("tracePower", false)
        ->setComment("Sample CPU (RAPL) and GPU (NVML) power as counter tracks.");
    desc.addUntracked<unsigned>("powerPeriodMs", 1000)
        ->setComment("Power sampling period in milliseconds (when tracePower is true).");
    desc.addUntracked<std::vector<std::string>>("traceModules", {})
        ->setComment("If non-empty, only trace these module labels (lower overhead, focused trace).");
    descriptions.add("PerfettoTraceService", desc);
  }

private:
  struct PerStream {
    unsigned sid = 0;
    bool in_event = false;
    bool source_open = false;
    unsigned long long eventId = 0;
  };

  bool tracing() const noexcept { return enabled_ && ::perfetto::TrackEvent::IsEnabled(); }

  // When traceModules is non-empty, only those module labels are traced (and only
  // their allocations/kernels are attributed). Empty = trace every module.
  bool selected(edm::ModuleDescription const& md) const {
    return traceModules_.empty() || std::binary_search(traceModules_.begin(), traceModules_.end(), md.moduleLabel());
  }

  bool withinBudget() noexcept { return maxEvents_ == 0 || seenEvents_.fetch_add(1) < maxEvents_; }

  // Publish (or clear) the thread-local "current module" used by the allocator
  // and GPU layers to attribute their work.
  // Push a module context on this thread's stack. |active| (== this module is
  // selected for tracing) controls whether work done inside it is attributed.
  static void pushContext(edm::StreamContext const& sc, edm::ModuleDescription const& md, bool active) noexcept {
    cms::perfetto::ModuleContext ctx;
    ctx.label = md.moduleLabel().c_str();
    ctx.type = md.moduleName().c_str();
    ctx.moduleId = md.id();
    ctx.streamId = sc.streamID().value();
    ctx.eventId = sc.eventID().event();
    ctx.active = active;
    cms::perfetto::pushModuleContext(ctx);
  }

  void preallocate(edm::service::SystemBounds const& bounds) {
    states_.assign(bounds.maxNumberOfStreams(), PerStream{});
    for (unsigned i = 0; i < states_.size(); ++i) {
      states_[i].sid = i;
      auto t = cms::perfetto::streamTrack(i);
      auto d = t.Serialize();
      d.set_name("edm::stream " + std::to_string(i));
      ::perfetto::TrackEvent::SetTrackDescriptor(t, d);
    }
  }

  // ----- Source (on the stream's lane for the reading thread) -----
  void preSourceEvent(edm::StreamID sid) {
    if (!tracing())
      return;
    auto& st = states_[sid.value()];
    if (!st.source_open) {
      st.source_open = true;
      TRACE_EVENT_BEGIN("cmssw.source", "Source", cms::perfetto::laneTrack(sid.value()), "stream", sid.value());
    }
  }

  void postSourceEvent(edm::StreamID sid) {
    if (!tracing())
      return;
    auto& st = states_[sid.value()];
    if (st.source_open) {
      TRACE_EVENT_END("cmssw.source", cms::perfetto::laneTrack(sid.value()));
      st.source_open = false;
    }
  }

  // ----- Per-stream event lifetime (serialized -> per-stream track) -----
  void preEvent(edm::StreamContext const& sc) {
    if (!tracing())
      return;
    auto& st = states_[sc.streamID().value()];
    if (st.in_event)
      return;
    if (!withinBudget()) {
      cms::perfetto::resetModuleContext();
      return;
    }
    st.in_event = true;
    auto const& id = sc.eventID();
    st.eventId = id.event();

    auto track = cms::perfetto::streamTrack(st.sid);
    TRACE_EVENT_BEGIN(
        "cmssw.event", "Event", track, "run", id.run(), "lumi", id.luminosityBlock(), "event", id.event());
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("run", "id", track), static_cast<double>(id.run()));
    TRACE_COUNTER(
        "cmssw.event", ::perfetto::CounterTrack("lumi", "id", track), static_cast<double>(id.luminosityBlock()));
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("event", "id", track), static_cast<double>(id.event()));
  }

  void preClearEvent(edm::StreamContext const& sc) {
    if (!tracing())
      return;
    if (!states_[sc.streamID().value()].in_event)
      return;
    TRACE_EVENT_BEGIN(
        "cmssw.cleanup", "Cleanup", cms::perfetto::laneTrack(sc.streamID().value()), "stream", sc.streamID().value());
  }

  void postClearEvent(edm::StreamContext const& sc) {
    if (!tracing())
      return;
    auto& st = states_[sc.streamID().value()];
    if (!st.in_event)
      return;
    TRACE_EVENT_END("cmssw.cleanup", cms::perfetto::laneTrack(sc.streamID().value()));
    TRACE_EVENT_END("cmssw.event", cms::perfetto::streamTrack(st.sid));
    st.in_event = false;
    emitThroughput();  // global events/s counter, from event-completion times
    cms::perfetto::resetModuleContext();
  }

  // Global "Throughput (events/s)" counter: a sliding-window rate over the last
  // kThroughputWindow event completions (across all streams), emitted on a single
  // process-level CounterTrack -- so any Perfetto UI shows the job's event rate
  // ramping up and reaching steady state, no UI plugin required.
  void emitThroughput() {
    timespec ts{};
    clock_gettime(CLOCK_BOOTTIME, &ts);
    int64_t const now = int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    double rate = 0.;
    {
      std::scoped_lock lock(throughputMutex_);
      completions_.push_back(now);
      while (completions_.size() > kThroughputWindow)
        completions_.pop_front();
      if (completions_.size() >= 2) {
        double const span_s = double(now - completions_.front()) / 1e9;
        if (span_s > 0.)
          rate = double(completions_.size() - 1) / span_s;
      }
    }
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("Throughput (events/s)"), rate);
  }

  // ----- Modules (per thread) -----
  void preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!tracing())
      return;
    if (!states_[sc.streamID().value()].in_event)
      return;
    auto const& md = *mcc.moduleDescription();
    bool const sel = selected(md);
    pushContext(sc, md, sel);  // pushed for every module so the stack stays balanced
    if (!sel)
      return;
    TRACE_EVENT_BEGIN("cmssw.module",
                      ::perfetto::DynamicString(md.moduleLabel()),
                      cms::perfetto::laneTrack(sc.streamID().value()),
                      "event",
                      sc.eventID().event(),
                      "module_id",
                      md.id(),
                      "cpp_type",
                      ::perfetto::DynamicString(md.moduleName()));
  }

  void postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!tracing())
      return;
    if (!states_[sc.streamID().value()].in_event)
      return;
    if (selected(*mcc.moduleDescription()))
      TRACE_EVENT_END("cmssw.module", cms::perfetto::laneTrack(sc.streamID().value()));
    cms::perfetto::popModuleContext();
  }

  void preModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!tracing())
      return;
    if (!states_[sc.streamID().value()].in_event)
      return;
    auto const& md = *mcc.moduleDescription();
    bool const sel = selected(md);
    pushContext(sc, md, sel);
    if (!sel)
      return;
    TRACE_EVENT_BEGIN("cmssw.acquire",
                      ::perfetto::DynamicString(md.moduleLabel()),
                      cms::perfetto::laneTrack(sc.streamID().value()),
                      "event",
                      sc.eventID().event(),
                      "cpp_type",
                      ::perfetto::DynamicString(md.moduleName()));
  }

  void postModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!tracing())
      return;
    if (!states_[sc.streamID().value()].in_event)
      return;
    if (selected(*mcc.moduleDescription()))
      TRACE_EVENT_END("cmssw.acquire", cms::perfetto::laneTrack(sc.streamID().value()));
    cms::perfetto::popModuleContext();
  }

  // ----- EventSetup modules (per thread; stream attribution is best-effort) -----
  static edm::StreamContext const* streamOf(edm::ESModuleCallingContext const& cc) {
    auto top = cc.getTopModuleCallingContext();
    if (!top || top->type() != edm::ParentContext::Type::kPlaceInPath)
      return nullptr;
    auto const* pip = top->parent().placeInPathContext();
    auto const* pc = pip ? pip->pathContext() : nullptr;
    return pc ? pc->streamContext() : nullptr;
  }

  void preESModule(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const& cc) {
    if (!tracing())
      return;
    auto const* sc = streamOf(cc);
    if (!sc || !states_[sc->streamID().value()].in_event)
      return;
    auto const* cd = cc.componentDescription();
    const char* name = (cd && !cd->label_.empty()) ? cd->label_.c_str() : (cd ? cd->type_.c_str() : "ESModule");
    TRACE_EVENT_BEGIN("cmssw.es",
                      ::perfetto::DynamicString(name),
                      cms::perfetto::laneTrack(sc->streamID().value()),
                      "stream",
                      sc->streamID().value());
  }

  void postESModule(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const& cc) {
    if (!tracing())
      return;
    auto const* sc = streamOf(cc);
    if (!sc || !states_[sc->streamID().value()].in_event)
      return;
    TRACE_EVENT_END("cmssw.es", cms::perfetto::laneTrack(sc->streamID().value()));
  }

  void postEndJob() {
    if (!enabled_ || !session_)
      return;
    // Stop observing allocations before the session is torn down: late frees
    // (e.g. from the AlpakaService destructor) must not touch a stopped session.
    if (traceAllocations_)
      cms::alpakatools::setCachingAllocatorMonitor(nullptr);
    // Stop the power sampler before the session is torn down (its thread emits).
    if (tracePower_)
      powerSampler_.stop();
    // Drain CUDA kernel activity into the still-open session, then stop it.
    if (traceGpuKernels_) {
      cuptiProfiler_.flush();
      cuptiProfiler_.stop();
    }
    ::perfetto::TrackEvent::Flush();
    session_->StopBlocking();
    auto trace_data = session_->ReadTraceBlocking();
    int fd = ::open(fileName_.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (fd >= 0) {
      [[maybe_unused]] auto n = ::write(fd, trace_data.data(), trace_data.size());
      ::close(fd);
    }
  }

  const bool enabled_;
  const std::string fileName_;
  const unsigned bufferSizeKB_;
  const unsigned shmemSizeKB_;
  const unsigned maxEvents_;
  const bool traceFunctions_;
  const bool traceAllocations_;
  const bool traceGpuKernels_;
  const bool tracePower_;
  const unsigned powerPeriodMs_;
  std::vector<std::string> traceModules_;

  std::unique_ptr<::perfetto::TracingSession> session_;

  std::vector<PerStream> states_;
  std::atomic<unsigned> seenEvents_{0};
  cms::perfetto::PerfettoAllocatorMonitor allocatorMonitor_;
  cms::perfetto::PerfettoCuptiProfiler cuptiProfiler_;
  cms::perfetto::PerfettoPowerSampler powerSampler_;

  static constexpr std::size_t kThroughputWindow = 16;  // events in the rate window
  std::mutex throughputMutex_;
  std::deque<int64_t> completions_;  // boottime ns of recent event completions
};

DEFINE_FWK_SERVICE(PerfettoTraceService);

// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocatorMonitor.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
#include "PerfTools/Perfetto/interface/PerfettoAllocatorMonitor.h"
#include "PerfTools/Perfetto/plugins/PerfettoCuptiProfiler.h"
#include "PerfTools/Perfetto/plugins/PerfettoPowerSampler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Records an in-process Perfetto trace (.pftrace) of a cmsRun job.
//
// Within one event, modules run concurrently on different threads (and an
// ExternalWork acquire() and produce() on different threads), so module,
// acquire, EventSetup, source and cleanup slices go on per-(stream, thread)
// lanes under the stream track (see CMSSWPerfettoLanes.h). The event lifetime,
// serialized per stream, goes on the stream track itself with run/lumi/event
// counters. Around every module call the service publishes a thread-local
// ModuleContext, used to attribute allocations and CMS_PERFETTO_FUNC slices.
class PerfettoTraceService {
public:
  PerfettoTraceService(edm::ParameterSet const& pset, edm::ActivityRegistry& ar)
      : fileName_(pset.getUntrackedParameter<std::string>("fileName")),
        maxEvents_(pset.getUntrackedParameter<unsigned>("maxEvents")),
        traceAllocations_(pset.getUntrackedParameter<bool>("traceAllocations")),
        traceModules_(pset.getUntrackedParameter<std::vector<std::string>>("traceModules")) {
    if (!pset.getUntrackedParameter<bool>("enabled"))
      return;
    std::ranges::sort(traceModules_);

    ::perfetto::TracingInitArgs args;
    args.backends = ::perfetto::kInProcessBackend;
    // A too small shared-memory buffer silently drops slices (TrackEvent uses the
    // kDrop policy, so tracing never blocks the framework threads). 32 KB is the
    // largest chunk size and minimizes contention between threads.
    args.shmem_size_hint_kb = pset.getUntrackedParameter<unsigned>("shmemSizeKB");
    args.shmem_page_size_hint_kb = 32;
    ::perfetto::Tracing::Initialize(args);
    cms::perfetto::TrackEvent::Register();

    bool const traceGpuKernels = pset.getUntrackedParameter<bool>("traceGpuKernels");
    bool const tracePower = pset.getUntrackedParameter<bool>("tracePower");
    ::perfetto::protos::gen::TrackEventConfig te;
    te.add_disabled_categories("*");
    for (const char* category :
         {"cmssw.event", "cmssw.source", "cmssw.module", "cmssw.acquire", "cmssw.es", "cmssw.cleanup"})
      te.add_enabled_categories(category);
    if (pset.getUntrackedParameter<bool>("traceFunctions"))
      te.add_enabled_categories("cmssw.func");
    if (traceAllocations_)
      te.add_enabled_categories("cmssw.alloc");
    if (traceGpuKernels)
      te.add_enabled_categories("cmssw.gpu");
    if (tracePower)
      te.add_enabled_categories("cmssw.power");

    ::perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(pset.getUntrackedParameter<unsigned>("bufferSizeKB"));
    auto* ds = cfg.add_data_sources()->mutable_config();
    ds->set_name("track_event");
    ds->set_track_event_config_raw(te.SerializeAsString());

    // The trace is kept in memory and written at the end of the job: the GPU
    // kernels, flushed at the end with their earlier device timestamps, then stay
    // correctly ordered.
    session_ = ::perfetto::Tracing::NewTrace();
    session_->Setup(cfg);
    session_->StartBlocking();

    auto process = ::perfetto::ProcessTrack::Current();
    auto desc = process.Serialize();
    desc.mutable_process()->set_process_name("cmsRun");
    cms::perfetto::TrackEvent::SetTrackDescriptor(process, desc);

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

    if (traceAllocations_)
      cms::alpakatools::setCachingAllocatorMonitor(&allocatorMonitor_);
    if (traceGpuKernels && !cuptiProfiler_.start())
      edm::LogWarning("PerfettoTraceService")
          << "traceGpuKernels: no CUDA device or CUPTI unavailable, not tracing GPU "
             "kernels";
    if (tracePower &&
        !powerSampler_.start(std::chrono::milliseconds(pset.getUntrackedParameter<unsigned>("powerPeriodMs"))))
      edm::LogWarning("PerfettoTraceService") << "tracePower: neither NVML nor a readable RAPL counter found, not "
                                                 "tracing power";
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("enabled", true)->setComment("Master switch; when false the service does nothing.");
    desc.addUntracked<std::string>("fileName", "cmsrun.pftrace")->setComment("Output Perfetto trace file.");
    desc.addUntracked<unsigned>("bufferSizeKB", 256 * 1024)
        ->setComment("In-memory trace buffer size in KB; bounds the trace size.");
    desc.addUntracked<unsigned>("shmemSizeKB", 64 * 1024)
        ->setComment("Producer shared-memory buffer size in KB; if too small, slices are silently dropped.");
    desc.addUntracked<unsigned>("maxEvents", 0)->setComment("Trace only the first maxEvents events (0: all).");
    desc.addUntracked<bool>("traceFunctions", false)
        ->setComment("Record the CMS_PERFETTO_FUNC/CMS_PERFETTO_SCOPE slices.");
    desc.addUntracked<bool>("traceAllocations", false)
        ->setComment("Record the alpaka caching-allocator transactions and memory counters.");
    desc.addUntracked<bool>("traceGpuKernels", false)->setComment("Record the CUDA kernels via CUPTI.");
    desc.addUntracked<bool>("tracePower", false)->setComment("Sample CPU (RAPL) and GPU (NVML) power.");
    desc.addUntracked<unsigned>("powerPeriodMs", 1000)->setComment("Power sampling period in ms.");
    desc.addUntracked<std::vector<std::string>>("traceModules", {})
        ->setComment("If not empty, trace only the modules with these labels.");
    descriptions.add("PerfettoTraceService", desc);
  }

private:
  // padded, as every module transition of the stream reads it
  struct alignas(64) StreamState {
    bool inEvent = false;
  };

  // Event rate over the last kSize event completions, from all streams.
  class RateWindow {
  public:
    void push(uint64_t t) { times_[n_++ % kSize] = t; }
    double rate() const {
      std::size_t const k = std::min<std::size_t>(n_, kSize);
      if (k < 2)
        return 0.;
      uint64_t const span = times_[(n_ - 1) % kSize] - times_[(n_ - k) % kSize];
      return span > 0 ? double(k - 1) * 1e9 / double(span) : 0.;
    }

  private:
    static constexpr std::size_t kSize = 16;
    std::array<uint64_t, kSize> times_{};
    std::size_t n_ = 0;
  };

  bool inEvent(edm::StreamID sid) const noexcept { return states_[sid.value()].inEvent; }

  bool selected(edm::ModuleDescription const& md) const {
    return traceModules_.empty() || std::ranges::binary_search(traceModules_, md.moduleLabel());
  }

  // Push the module context; true if the module's slice must be recorded.
  bool enterModule(edm::StreamContext const& sc, edm::ModuleDescription const& md) const {
    bool const traced = selected(md);
    cms::perfetto::ModuleContext ctx;
    if (traced) {
      ctx.label = md.moduleLabel().c_str();
      ctx.streamId = sc.streamID().value();
    }
    // pushed for untraced modules too, so that their work is not attributed to
    // the module they may be nested in
    return cms::perfetto::pushModuleContext(ctx) && traced;
  }

  // Pop the module context; true if the module's slice was recorded.
  static bool exitModule() { return cms::perfetto::popModuleContext().label != nullptr; }

  // The stream for which an EventSetup module runs, if any.
  static edm::StreamContext const* streamOf(edm::ESModuleCallingContext const& cc) {
    auto const* top = cc.getTopModuleCallingContext();
    if (!top)
      return nullptr;
    switch (top->type()) {
      case edm::ParentContext::Type::kPlaceInPath:
        return top->placeInPathContext()->pathContext()->streamContext();
      case edm::ParentContext::Type::kStream:
        return top->streamContext();
      default:
        return nullptr;
    }
  }

  void preallocate(edm::service::SystemBounds const& bounds) {
    states_ = std::vector<StreamState>(bounds.maxNumberOfStreams());
    for (unsigned sid = 0; sid < states_.size(); ++sid) {
      auto const track = cms::perfetto::streamTrack(sid);
      auto desc = track.Serialize();
      desc.set_name("edm::stream " + std::to_string(sid));
      cms::perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    }
  }

  void preSourceEvent(edm::StreamID sid) {
    TRACE_EVENT_BEGIN("cmssw.source", "Source", cms::perfetto::laneTrack(sid.value()), "stream", sid.value());
  }

  void postSourceEvent(edm::StreamID sid) { TRACE_EVENT_END("cmssw.source", cms::perfetto::laneTrack(sid.value())); }

  void preEvent(edm::StreamContext const& sc) {
    if (!cms::perfetto::TrackEvent::IsEnabled())
      return;
    if (maxEvents_ > 0 && seenEvents_.fetch_add(1, std::memory_order_relaxed) >= maxEvents_)
      return;
    // Seed the rate window with the first event start: all streams complete their
    // first, slow event nearly together, and completions alone would report a
    // spuriously high rate at startup.
    if (!throughputSeeded_.exchange(true, std::memory_order_relaxed)) {
      std::scoped_lock lock(throughputMutex_);
      throughput_.push(cms::perfetto::TrackEvent::GetTraceTimeNs());
    }
    states_[sc.streamID().value()].inEvent = true;

    auto const& id = sc.eventID();
    auto const track = cms::perfetto::streamTrack(sc.streamID().value());
    TRACE_EVENT_BEGIN(
        "cmssw.event", "Event", track, "run", id.run(), "lumi", id.luminosityBlock(), "event", id.event());
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("run", "id", track), double(id.run()));
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("lumi", "id", track), double(id.luminosityBlock()));
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("event", "id", track), double(id.event()));
  }

  void preClearEvent(edm::StreamContext const& sc) {
    if (!inEvent(sc.streamID()))
      return;
    TRACE_EVENT_BEGIN("cmssw.cleanup", "Cleanup", cms::perfetto::laneTrack(sc.streamID().value()));
  }

  void postClearEvent(edm::StreamContext const& sc) {
    auto& state = states_[sc.streamID().value()];
    if (!state.inEvent)
      return;
    TRACE_EVENT_END("cmssw.cleanup", cms::perfetto::laneTrack(sc.streamID().value()));
    TRACE_EVENT_END("cmssw.event", cms::perfetto::streamTrack(sc.streamID().value()));
    state.inEvent = false;

    double rate;
    {
      std::scoped_lock lock(throughputMutex_);
      throughput_.push(cms::perfetto::TrackEvent::GetTraceTimeNs());
      rate = throughput_.rate();
    }
    TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("Throughput (events/s)"), rate);
  }

  // Module labels and C++ types are owned by the ModuleDescription and outlive the
  // session: StaticString lets perfetto intern them instead of copying them into
  // every slice.
  void preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!inEvent(sc.streamID()))
      return;
    auto const& md = *mcc.moduleDescription();
    if (!enterModule(sc, md))
      return;
    TRACE_EVENT_BEGIN("cmssw.module",
                      ::perfetto::StaticString(md.moduleLabel().c_str()),
                      cms::perfetto::laneTrack(sc.streamID().value()),
                      "event",
                      sc.eventID().event(),
                      "module_id",
                      md.id(),
                      "cpp_type",
                      ::perfetto::DynamicString(md.moduleName()));
  }

  void postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const&) {
    if (inEvent(sc.streamID()) && exitModule())
      TRACE_EVENT_END("cmssw.module", cms::perfetto::laneTrack(sc.streamID().value()));
  }

  void preModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
    if (!inEvent(sc.streamID()))
      return;
    auto const& md = *mcc.moduleDescription();
    if (!enterModule(sc, md))
      return;
    TRACE_EVENT_BEGIN("cmssw.acquire",
                      ::perfetto::StaticString(md.moduleLabel().c_str()),
                      cms::perfetto::laneTrack(sc.streamID().value()),
                      "event",
                      sc.eventID().event(),
                      "cpp_type",
                      ::perfetto::DynamicString(md.moduleName()));
  }

  void postModuleEventAcquire(edm::StreamContext const& sc, edm::ModuleCallingContext const&) {
    if (inEvent(sc.streamID()) && exitModule())
      TRACE_EVENT_END("cmssw.acquire", cms::perfetto::laneTrack(sc.streamID().value()));
  }

  void preESModule(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const& cc) {
    auto const* sc = streamOf(cc);
    if (!sc || !inEvent(sc->streamID()))
      return;
    auto const* cd = cc.componentDescription();
    const char* name = !cd ? "ESModule" : (cd->label_.empty() ? cd->type_.c_str() : cd->label_.c_str());
    TRACE_EVENT_BEGIN("cmssw.es",
                      ::perfetto::StaticString(name),
                      cms::perfetto::laneTrack(sc->streamID().value()),
                      "stream",
                      sc->streamID().value());
  }

  void postESModule(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const& cc) {
    auto const* sc = streamOf(cc);
    if (sc && inEvent(sc->streamID()))
      TRACE_EVENT_END("cmssw.es", cms::perfetto::laneTrack(sc->streamID().value()));
  }

  void postEndJob() {
    // stop every producer before the session goes away; later frees (e.g. from
    // the AlpakaService destructor) must not reach it
    if (traceAllocations_)
      cms::alpakatools::setCachingAllocatorMonitor(nullptr);
    powerSampler_.stop();
    cuptiProfiler_.stop();  // drains the pending kernel records into the session

    cms::perfetto::TrackEvent::Flush();
    session_->StopBlocking();
    std::vector<char> const trace = session_->ReadTraceBlocking();
    session_.reset();

    std::ofstream out(fileName_, std::ios::binary | std::ios::trunc);
    out.write(trace.data(), trace.size());
    if (out.flush())
      edm::LogInfo("PerfettoTraceService") << "Wrote " << trace.size() << " bytes to " << fileName_;
    else
      edm::LogError("PerfettoTraceService") << "Failed to write the trace to " << fileName_;
  }

  std::string const fileName_;
  unsigned const maxEvents_;
  bool const traceAllocations_;
  std::vector<std::string> traceModules_;  // sorted

  std::unique_ptr<::perfetto::TracingSession> session_;
  std::vector<StreamState> states_;
  std::atomic<unsigned> seenEvents_{0};

  std::mutex throughputMutex_;
  std::atomic<bool> throughputSeeded_{false};
  RateWindow throughput_;

  cms::perfetto::PerfettoAllocatorMonitor allocatorMonitor_;
  cms::perfetto::PerfettoCuptiProfiler cuptiProfiler_;
  cms::perfetto::PerfettoPowerSampler powerSampler_;
};

DEFINE_FWK_SERVICE(PerfettoTraceService);

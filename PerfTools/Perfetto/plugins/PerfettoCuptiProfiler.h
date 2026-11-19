// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_plugins_PerfettoCuptiProfiler_h
#define PerfTools_Perfetto_plugins_PerfettoCuptiProfiler_h

// PERFETTO_HAS_CUPTI is defined by the plugin BuildFile only where CUDA exists;
// elsewhere a no-op stub keeps the service building (traceGpuKernels does nothing).
#ifdef PERFETTO_HAS_CUPTI

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"

#include <cuda_runtime.h>
#include <cupti.h>

#include <cxxabi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace cms::perfetto {

  // Records CUDA kernels through the CUPTI activity API, as slices on one track
  // per (device, CUDA stream) at their device-side start/end times. Each slice
  // carries the launch's resource usage (registers, shared/local memory), its
  // grid/block, an occupancy estimate and the CUPTI correlation id.
  // Optionally also the memory copies and memsets, on the same tracks, with their
  // direction, size, source/destination memory kind and bandwidth.
  // Optionally the records carry the label of the module that issued them: a CUPTI
  // callback on the launch / copy / memset / allocation driver calls records, on the
  // issuing thread, the current module context under the correlation id of the call.
  // Optionally also every CUDA allocation and free (cudaMalloc, cudaHostAlloc,
  // cudaMallocAsync, ... also those of the alpaka caching allocator, which uses them
  // underneath), as instants. There is no running-total counter: CUPTI delivers its
  // buffers out of time order, so a total summed on arrival would be wrong; sum the
  // instants in time order instead.
  // CUPTI works at driver level: release plugins need no rebuild, and kernels are
  // not serialized.
  class PerfettoCuptiProfiler {
  public:
    PerfettoCuptiProfiler() = default;
    // Stop CUPTI and drop the records that are still buffered.
    ~PerfettoCuptiProfiler() {
      s_instance.store(nullptr);
      halt();
    }
    PerfettoCuptiProfiler(PerfettoCuptiProfiler const&) = delete;
    PerfettoCuptiProfiler& operator=(PerfettoCuptiProfiler const&) = delete;

    // True if a CUDA device is present and CUPTI records at least one of the requested
    // kinds of activity. With modules, the records also carry the issuing module.
    bool start(bool kernels = true, bool copies = false, bool memory = false, bool modules = true) {
      int count = 0;
      if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return false;
      props_.resize(count);
      for (int d = 0; d < count; ++d)
        cudaGetDeviceProperties(&props_[d], d);

      // offset from the CUPTI clock to the trace clock
      uint64_t cuptiNow = 0;
      cuptiGetTimestamp(&cuptiNow);
      offsetNs_ = int64_t(TrackEvent::GetTraceTimeNs()) - int64_t(cuptiNow);
      clockId_ = static_cast<uint32_t>(TrackEvent::GetTraceClockId());

      if (cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted) != CUPTI_SUCCESS)
        return false;
      if (modules)
        subscribe(kernels, copies, memory);
      s_instance.store(this);
      if (kernels)
        enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, "kernels");
      if (copies) {
        enable(CUPTI_ACTIVITY_KIND_MEMCPY, "memory copies");
        enable(CUPTI_ACTIVITY_KIND_MEMCPY2, "memory copies between devices");
        enable(CUPTI_ACTIVITY_KIND_MEMSET, "memsets");
      }
      if (memory)
        enable(CUPTI_ACTIVITY_KIND_MEMORY2, "allocations and frees");
      active_ = !enabled_.empty();
      if (!active_)
        halt();
      return active_;
    }

    // Stop recording and drain the buffered records into the (still open) session.
    void stop() {
      halt();
      if (auto const lost = lost_.exchange(0); lost > 0)
        edm::LogWarning("PerfettoTraceService")
            << "traceGpuModules: " << lost << " GPU records have no module, because more than " << kOwnerSlots
            << " CUDA calls were issued before CUPTI delivered them";
    }

  private:
    // Disable the activity kinds and the subscriber, then flush the buffered records.
    void halt() {
      for (auto kind : enabled_)
        cuptiActivityDisable(kind);
      enabled_.clear();
      if (subscribed_) {
        cuptiUnsubscribe(subscriber_);
        subscribed_ = false;
      }
      if (active_) {
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        active_ = false;
      }
      s_instance.store(nullptr);
    }

    static constexpr uint64_t kGpuTrackBase = 0x4750550000000000ull;  // "GPU"
    static constexpr std::size_t kBufferSize = 8 * 1024 * 1024;
    // One slot per correlation id modulo kOwnerSlots: the module of a call stays
    // available until kOwnerSlots later calls are issued.
    static constexpr std::size_t kOwnerSlots = std::size_t{1} << 22;
    static constexpr unsigned kMaxModules = 1u << 16;

    static const char* resultString(CUptiResult status) {
      const char* text = nullptr;
      cuptiGetResultString(status, &text);
      return text ? text : "unknown error";
    }

    // A kind that CUPTI refuses is reported and does not stop the other kinds.
    void enable(CUpti_ActivityKind kind, const char* what) {
      if (auto const status = cuptiActivityEnable(kind); status == CUPTI_SUCCESS)
        enabled_.push_back(kind);
      else
        edm::LogWarning("PerfettoTraceService")
            << "CUPTI does not record the CUDA " << what << ": " << resultString(status);
    }

    // Callbacks on the driver calls that issue the recorded device work. Only one
    // CUPTI subscriber can exist per process (e.g. not under nsys): without it the
    // records have no module.
    void subscribe(bool kernels, bool copies, bool memory) {
      owners_ = std::make_unique<std::atomic<uint64_t>[]>(kOwnerSlots);
      labels_ = std::make_unique<std::atomic<const char*>[]>(kMaxModules);
      if (auto const status = cuptiSubscribe(&subscriber_, apiCallback, this); status != CUPTI_SUCCESS) {
        edm::LogWarning("PerfettoTraceService") << "traceGpuModules: CUPTI does not accept a subscriber ("
                                                << resultString(status) << "), the GPU activity has no module";
        owners_.reset();
        labels_.reset();
        return;
      }
      subscribed_ = true;
      for (int cbid = 1; cbid < CUPTI_DRIVER_TRACE_CBID_SIZE; ++cbid) {
        const char* name = nullptr;
        if (cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, cbid, &name) != CUPTI_SUCCESS || !name)
          continue;
        std::string_view const n(name);
        // cuLaunchHostFunc gives no kernel record
        if ((kernels &&
             ((n.starts_with("cuLaunch") && !n.starts_with("cuLaunchHostFunc")) || n.starts_with("cuGraphLaunch"))) ||
            (copies && (n.starts_with("cuMemcpy") || n.starts_with("cuMemset"))) ||
            (memory && (n.starts_with("cuMemAlloc") || n.starts_with("cuMemHostAlloc") || n.starts_with("cuMemFree"))))
          cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid);
      }
    }

    // Index of the module of this context in labels_, plus one; 0 for no module.
    // A slot of labels_ is written once, so an index always gives the same label.
    uint32_t moduleIndex(ModuleContext const& ctx) {
      if (!ctx.label || ctx.moduleId >= kMaxModules)
        return 0;
      auto& slot = labels_[ctx.moduleId];
      const char* seen = slot.load(std::memory_order_acquire);
      if (!seen && slot.compare_exchange_strong(seen, ctx.label))
        return ctx.moduleId + 1;
      return seen == ctx.label ? ctx.moduleId + 1 : 0;
    }

    // Runs on the thread that issues the driver call, inside the call: it takes no
    // lock and allocates nothing. One atomic word holds the correlation id and the
    // module, so a reader cannot pair an id with the module of another call.
    static void apiCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId, const void* cbdata) {
      if (domain != CUPTI_CB_DOMAIN_DRIVER_API)
        return;
      auto const* info = static_cast<CUpti_CallbackData const*>(cbdata);
      if (info->callbackSite != CUPTI_API_ENTER)
        return;
      auto* self = static_cast<PerfettoCuptiProfiler*>(userdata);
      uint64_t const entry = uint64_t{info->correlationId} << 32 | self->moduleIndex(currentModuleContext());
      self->owners_[info->correlationId % kOwnerSlots].store(entry, std::memory_order_release);
    }

    // The module that issued the work with this correlation id. "" for the framework,
    // for an untraced module, and when a later call has taken the slot. The entry
    // stays: all the records of one cuGraphLaunch share its correlation id.
    const char* owner(uint32_t correlationId) {
      if (!owners_)
        return "";
      uint64_t const entry = owners_[correlationId % kOwnerSlots].load(std::memory_order_acquire);
      auto const id = static_cast<uint32_t>(entry >> 32);
      auto const module = static_cast<uint32_t>(entry);
      if (id != correlationId) {
        if (static_cast<int32_t>(id - correlationId) > 0)
          lost_.fetch_add(1, std::memory_order_relaxed);
        return "";
      }
      return module == 0 ? "" : labels_[module - 1].load(std::memory_order_acquire);
    }

    // CUPTI gives 0 for both timestamps of a record when it did not collect them.
    static bool hasTimestamps(uint64_t start, uint64_t end) { return start != 0 || end != 0; }

    static void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
      *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, kBufferSize));
      *size = *buffer ? kBufferSize : 0;
      *maxNumRecords = 0;
    }

    static void bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
      if (auto* self = s_instance.load(); self && validSize > 0) {
        CUpti_Activity* record = nullptr;
        while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
          if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
            self->handleKernel(reinterpret_cast<CUpti_ActivityKernel9 const*>(record));
          else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY)
            self->handleMemcpy(reinterpret_cast<CUpti_ActivityMemcpy7 const*>(record));
          else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY2)  // copy involving another device's context
            self->handleMemcpy(reinterpret_cast<CUpti_ActivityMemcpyPtoP4 const*>(record));
          else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET)
            self->handleMemset(reinterpret_cast<CUpti_ActivityMemset5 const*>(record));
          else if (record->kind == CUPTI_ACTIVITY_KIND_MEMORY2)
            self->handleMemory(reinterpret_cast<CUpti_ActivityMemory4 const*>(record));
        }
      }
      std::free(buffer);
    }

    ::perfetto::Track gpuTrack(uint32_t device, uint32_t stream) {
      ::perfetto::Track const track(kGpuTrackBase | (uint64_t{device} << 32) | stream,
                                    ::perfetto::ProcessTrack::Current());
      std::scoped_lock lock(mutex_);
      if (named_.insert(track.uuid).second) {
        auto desc = track.Serialize();
        desc.set_name("GPU" + std::to_string(device) + " stream " + std::to_string(stream));
        TrackEvent::SetTrackDescriptor(track, desc);
      }
      return track;
    }

    // The user functor of an alpaka gpuKernel<Functor, ...> wrapper, else the
    // name capped at 96 characters.
    static std::string_view shortName(std::string_view name) {
      constexpr std::string_view kWrapper = "gpuKernel<";
      auto const p = name.find(kWrapper);
      if (p == std::string_view::npos)
        return name.substr(0, 96);
      name.remove_prefix(p + kWrapper.size());
      int depth = 0;
      for (std::size_t i = 0; i < name.size(); ++i) {
        char const c = name[i];
        if (c == '<')
          ++depth;
        else if ((c == '>' || c == ',') && depth == 0)
          return name.substr(0, i);
        else if (c == '>')
          --depth;
      }
      return name;
    }

    // Theoretical occupancy from the thread, register and shared-memory limits
    // per SM (ignores allocation granularity).
    double occupancy(uint32_t device, uint32_t regsPerThread, uint32_t smem, uint32_t blockThreads) const {
      if (device >= props_.size() || blockThreads == 0)
        return 0.;
      auto const& p = props_[device];
      int const byThreads = p.maxThreadsPerMultiProcessor / int(blockThreads);
      int const byRegs = regsPerThread > 0 ? p.regsPerMultiprocessor / int(regsPerThread * blockThreads) : byThreads;
      int const bySmem = smem > 0 ? int(p.sharedMemPerMultiprocessor / smem) : byThreads;
      int const blocks = std::max(0, std::min({byThreads, byRegs, bySmem}));
      return double(blocks * int(blockThreads)) / double(p.maxThreadsPerMultiProcessor);
    }

    void handleKernel(CUpti_ActivityKernel9 const* k) {
      if (!hasTimestamps(k->start, k->end))
        return;
      const char* mangled = k->name ? k->name : "";
      char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, nullptr);
      std::string const full = demangled ? demangled : mangled;
      std::free(demangled);
      std::string const name(shortName(full));

      uint32_t const blockThreads = k->blockX * k->blockY * k->blockZ;
      double const occ =
          occupancy(k->deviceId, k->registersPerThread, k->staticSharedMemory + k->dynamicSharedMemory, blockThreads);
      std::string const grid =
          std::to_string(k->gridX) + "x" + std::to_string(k->gridY) + "x" + std::to_string(k->gridZ);
      std::string const block =
          std::to_string(k->blockX) + "x" + std::to_string(k->blockY) + "x" + std::to_string(k->blockZ);

      auto const track = gpuTrack(k->deviceId, k->streamId);
      ::perfetto::TraceTimestamp const begin{clockId_, uint64_t(int64_t(k->start) + offsetNs_)};
      ::perfetto::TraceTimestamp const end{clockId_, uint64_t(int64_t(k->end) + offsetNs_)};

      TRACE_EVENT_BEGIN("cmssw.gpu",
                        ::perfetto::DynamicString(name),
                        track,
                        begin,
                        "registers_per_thread",
                        k->registersPerThread,
                        "static_smem_B",
                        k->staticSharedMemory,
                        "dynamic_smem_B",
                        k->dynamicSharedMemory,
                        "local_per_thread_B",
                        k->localMemoryPerThread,
                        "local_total_B",
                        k->localMemoryTotal,
                        "grid",
                        ::perfetto::DynamicString(grid),
                        "block",
                        ::perfetto::DynamicString(block),
                        "occupancy_est",
                        occ,
                        "correlation_id",
                        k->correlationId,
                        "module",
                        owner(k->correlationId),
                        "kernel",
                        ::perfetto::DynamicString(full));
      TRACE_EVENT_END("cmssw.gpu", track, end);
    }

    static bool isHost(uint32_t memoryKind) {
      return memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE || memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_PINNED;
    }

    static bool isDevice(uint32_t memoryKind) {
      return memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE || memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_ARRAY ||
             memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC;
    }

    // A MEMCPY2 record has copyKind PTOP also for a host <-> device copy: take the
    // direction from the memory kinds when both are host or device memory.
    static const char* copyName(uint8_t kind, uint8_t src, uint8_t dst) {
      if (kind == CUPTI_ACTIVITY_MEMCPY_KIND_PTOP) {
        if (isHost(src) && isDevice(dst))
          kind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
        else if (isDevice(src) && isHost(dst))
          kind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH;
        else if (isHost(src) && isHost(dst))
          kind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOH;
      }
      switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
          return "memcpy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
          return "memcpy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
          return "memcpy DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
          return "memcpy HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
          return "memcpy PtoP";
        default:
          return "memcpy other";
      }
    }

    // pageable host memory makes a copy synchronous with respect to the host
    static const char* memoryKind(uint8_t kind) {
      switch (kind) {
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
          return "pageable";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
          return "pinned";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
          return "device";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
          return "array";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
          return "managed";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
          return "device static";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
          return "managed static";
        case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
          return "unknown";
        default:
          return "other";
      }
    }

    static double bandwidthGBps(uint64_t bytes, uint64_t start, uint64_t end) {
      return end > start ? double(bytes) / double(end - start) : 0.;  // B/ns = GB/s
    }

    // CUpti_ActivityMemcpy7, or CUpti_ActivityMemcpyPtoP4: with several devices, a copy
    // from/to memory of another device's context (e.g. pinned host memory allocated
    // while another device was current) is reported as MEMCPY2
    template <typename Record>
    void handleMemcpy(Record const* c) {
      if (!hasTimestamps(c->start, c->end))
        return;
      auto const track = gpuTrack(c->deviceId, c->streamId);
      TRACE_EVENT_BEGIN("cmssw.gpu.copy",
                        ::perfetto::StaticString(copyName(c->copyKind, c->srcKind, c->dstKind)),
                        track,
                        ::perfetto::TraceTimestamp{clockId_, uint64_t(int64_t(c->start) + offsetNs_)},
                        "bytes",
                        c->bytes,
                        "src",
                        ::perfetto::StaticString(memoryKind(c->srcKind)),
                        "dst",
                        ::perfetto::StaticString(memoryKind(c->dstKind)),
                        "bandwidth_GBps",
                        bandwidthGBps(c->bytes, c->start, c->end),
                        "correlation_id",
                        c->correlationId,
                        "module",
                        owner(c->correlationId));
      TRACE_EVENT_END(
          "cmssw.gpu.copy", track, ::perfetto::TraceTimestamp{clockId_, uint64_t(int64_t(c->end) + offsetNs_)});
    }

    void handleMemset(CUpti_ActivityMemset5 const* m) {
      if (!hasTimestamps(m->start, m->end))
        return;
      auto const track = gpuTrack(m->deviceId, m->streamId);
      TRACE_EVENT_BEGIN("cmssw.gpu.copy",
                        "memset",
                        track,
                        ::perfetto::TraceTimestamp{clockId_, uint64_t(int64_t(m->start) + offsetNs_)},
                        "bytes",
                        m->bytes,
                        "dst",
                        ::perfetto::StaticString(memoryKind(uint8_t(m->memoryKind))),
                        "bandwidth_GBps",
                        bandwidthGBps(m->bytes, m->start, m->end),
                        "correlation_id",
                        m->correlationId,
                        "module",
                        owner(m->correlationId));
      TRACE_EVENT_END(
          "cmssw.gpu.copy", track, ::perfetto::TraceTimestamp{clockId_, uint64_t(int64_t(m->end) + offsetNs_)});
    }

    // allocation / free instants on a "CUDA memory <space>" track: "host" for pageable
    // and pinned host memory, "dev<d>" for every other kind
    void handleMemory(CUpti_ActivityMemory4 const* m) {
      bool const alloc = m->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION;
      if (!alloc && m->memoryOperationType != CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE)
        return;
      bool const device = !isHost(m->memoryKind);
      ::perfetto::Track const track(kMemoryTrackBase | (device ? m->deviceId : 0xffffu),
                                    ::perfetto::ProcessTrack::Current());
      {
        std::scoped_lock lock(mutex_);
        if (named_.insert(track.uuid).second) {
          auto desc = track.Serialize();
          desc.set_name(device ? "CUDA memory dev" + std::to_string(m->deviceId) : std::string("CUDA memory host"));
          TrackEvent::SetTrackDescriptor(track, desc);
        }
      }
      TRACE_EVENT_INSTANT("cmssw.gpu.memory",
                          ::perfetto::StaticString(alloc ? "cuda alloc" : "cuda free"),
                          track,
                          ::perfetto::TraceTimestamp{clockId_, uint64_t(int64_t(m->timestamp) + offsetNs_)},
                          "bytes",
                          m->bytes,
                          "kind",
                          ::perfetto::StaticString(memoryKind(uint8_t(m->memoryKind))),
                          "async",
                          m->isAsync != 0,
                          "pool_B",
                          m->memoryPoolConfig.memoryPoolType == CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL
                              ? m->memoryPoolConfig.pool.size
                              : uint64_t(0),
                          "device",
                          device ? int(m->deviceId) : -1,
                          "module",
                          owner(m->correlationId));
    }

    static constexpr uint64_t kMemoryTrackBase = 0x4d454d0000000000ull;  // "MEM"

    bool active_ = false;
    bool subscribed_ = false;
    std::vector<CUpti_ActivityKind> enabled_;
    CUpti_SubscriberHandle subscriber_ = nullptr;
    // (correlation id << 32 | index in labels_ + 1) of the last call issued with this id modulo kOwnerSlots
    std::unique_ptr<std::atomic<uint64_t>[]> owners_;
    std::unique_ptr<std::atomic<const char*>[]> labels_;  // module id -> module label
    std::atomic<uint64_t> lost_{0};                       // records whose slot holds a later call
    int64_t offsetNs_ = 0;                                // CUPTI clock -> trace clock
    uint32_t clockId_ = 0;
    std::vector<cudaDeviceProp> props_;
    std::mutex mutex_;  // CUPTI may deliver buffers from several threads
    std::unordered_set<uint64_t> named_;

    // read on CUPTI threads
    static inline std::atomic<PerfettoCuptiProfiler*> s_instance{nullptr};
  };

}  // namespace cms::perfetto

#else  // PERFETTO_HAS_CUPTI

namespace cms::perfetto {

  class PerfettoCuptiProfiler {
  public:
    bool start(bool = true, bool = false, bool = false, bool = true) { return false; }
    void stop() {}
  };

}  // namespace cms::perfetto

#endif  // PERFETTO_HAS_CUPTI

#endif  // PerfTools_Perfetto_plugins_PerfettoCuptiProfiler_h

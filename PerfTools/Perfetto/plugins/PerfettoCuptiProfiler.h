// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_plugins_PerfettoCuptiProfiler_h
#define PerfTools_Perfetto_plugins_PerfettoCuptiProfiler_h

// PERFETTO_HAS_CUPTI is defined by the plugin BuildFile only where CUDA exists;
// elsewhere a no-op stub keeps the service building (traceGpuKernels does nothing).
#ifdef PERFETTO_HAS_CUPTI

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"

#include <cuda_runtime.h>
#include <cupti.h>

#include <cxxabi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
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
  // CUPTI works at driver level: release plugins need no rebuild, and kernels are
  // not serialized.
  class PerfettoCuptiProfiler {
  public:
    // True if a CUDA device is present and CUPTI accepted the configuration.
    bool start() {
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

      s_instance.store(this);
      active_ = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted) == CUPTI_SUCCESS &&
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS;
      if (!active_)
        s_instance.store(nullptr);
      return active_;
    }

    // Stop recording and drain the buffered records into the (still open) session.
    void stop() {
      if (active_) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        active_ = false;
      }
      s_instance.store(nullptr);
    }

  private:
    static constexpr uint64_t kGpuTrackBase = 0x4750550000000000ull;  // "GPU"
    static constexpr std::size_t kBufferSize = 8 * 1024 * 1024;

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
                        "kernel",
                        ::perfetto::DynamicString(full));
      TRACE_EVENT_END("cmssw.gpu", track, end);
    }

    bool active_ = false;
    int64_t offsetNs_ = 0;  // CUPTI clock -> trace clock
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
    bool start() { return false; }
    void stop() {}
  };

}  // namespace cms::perfetto

#endif  // PERFETTO_HAS_CUPTI

#endif  // PerfTools_Perfetto_plugins_PerfettoCuptiProfiler_h

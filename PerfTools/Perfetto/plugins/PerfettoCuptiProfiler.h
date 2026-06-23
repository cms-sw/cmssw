// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#pragma once

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include <perfetto.h>

// CUDA/CUPTI exist only on some architectures, so the plugin BuildFile pulls in
// cuda/cupti and defines PERFETTO_HAS_CUPTI only inside <iftool name="cuda">.
// Build the real profiler when it is set; otherwise provide a no-op stub below so
// PerfettoTraceService compiles unchanged (traceGpuKernels then does nothing).
#ifdef PERFETTO_HAS_CUPTI

#include <cuda_runtime.h>
#include <cupti.h>

#include <cxxabi.h>
#include <time.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

// Streams CUDA kernel activity into the running Perfetto session using CUPTI.
//
// Each kernel is emitted as a slice on a per-(device, CUDA stream) track, placed
// at the *device-side* start/end timestamps reported by CUPTI -- i.e. the real
// GPU execution time, not the host enqueue/wait time. The slice is annotated with
// the per-launch static resource usage CUPTI reports (registers/thread, static &
// dynamic shared memory, local memory/thread), the launch configuration
// (grid/block), an estimated theoretical occupancy, and the CUPTI correlation id
// (which links it to the host module that launched it).
//
// CUPTI is a driver-level profiler, so this works on the already-compiled release
// Alpaka/CUDA plugins without rebuilding them. The activity-tracing path is low
// overhead and does not serialize kernels.
namespace cms::perfetto {

  class PerfettoCuptiProfiler {
  public:
    // Returns true if CUPTI kernel tracing was activated (a CUDA device is present
    // and CUPTI accepted the configuration).
    bool start() {
      int count = 0;
      if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return false;

      // Cache per-device properties for the occupancy estimate.
      props_.resize(count);
      for (int d = 0; d < count; ++d)
        cudaGetDeviceProperties(&props_[d], d);

      // Correlate the CUPTI clock to CLOCK_BOOTTIME (Perfetto's default trace
      // clock), so GPU slices line up with the host timeline.
      uint64_t cuptiNow = 0;
      cuptiGetTimestamp(&cuptiNow);
      timespec ts{};
      clock_gettime(CLOCK_BOOTTIME, &ts);
      int64_t bootNow = int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
      offsetNs_ = bootNow - int64_t(cuptiNow);

      s_instance = this;
      if (cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted) != CUPTI_SUCCESS)
        return false;
      if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS)
        return false;
      active_ = true;
      return true;
    }

    // Drain remaining activity records into the (still open) Perfetto session.
    void flush() {
      if (active_)
        cuptiActivityFlushAll(1);
    }

    void stop() {
      if (active_) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        active_ = false;
      }
      s_instance = nullptr;
    }

  private:
    static constexpr uint64_t kGpuTrackBase = 0x4750550000000000ull;  // "GPU....."

    static void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
      constexpr size_t kSize = 8 * 1024 * 1024;
      *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, kSize));
      *size = kSize;
      *maxNumRecords = 0;
    }

    static void bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
      if (s_instance && validSize > 0) {
        CUpti_Activity* record = nullptr;
        while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
          if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
            s_instance->handleKernel(reinterpret_cast<CUpti_ActivityKernel9*>(record));
        }
      }
      std::free(buffer);
    }

    ::perfetto::Track gpuTrack(uint32_t device, uint32_t stream) {
      uint64_t const uuid = kGpuTrackBase | (uint64_t{device} << 24) | stream;
      {
        std::scoped_lock lock(mutex_);
        if (seen_.insert(uuid).second) {
          ::perfetto::Track t(uuid, ::perfetto::ProcessTrack::Current());
          auto d = t.Serialize();
          d.set_name("GPU" + std::to_string(device) + " stream " + std::to_string(stream));
          ::perfetto::TrackEvent::SetTrackDescriptor(t, d);
        }
      }
      return ::perfetto::Track(uuid, ::perfetto::ProcessTrack::Current());
    }

    static std::string shorten(const char* demangled) {
      std::string s = demangled ? demangled : "";
      auto p = s.find("gpuKernel<");
      if (p == std::string::npos)
        return s.size() > 96 ? s.substr(0, 96) : s;
      p += 10;
      int depth = 0;
      size_t i = p;
      for (; i < s.size(); ++i) {
        char c = s[i];
        if (c == '<')
          ++depth;
        else if (c == '>') {
          if (depth == 0)
            break;
          --depth;
        } else if (c == ',' && depth == 0)
          break;
      }
      return s.substr(p, i - p);
    }

    double occupancy(int device, uint32_t regsPerThread, uint32_t smem, uint32_t blockThreads) const {
      if (device < 0 || device >= int(props_.size()) || blockThreads == 0)
        return 0.;
      auto const& p = props_[device];
      int blocksThreads = p.maxThreadsPerMultiProcessor / int(blockThreads);
      int blocksReg = regsPerThread > 0 ? p.regsPerMultiprocessor / int(regsPerThread * blockThreads) : blocksThreads;
      int blocksSmem = smem > 0 ? int(p.sharedMemPerMultiprocessor / smem) : blocksThreads;
      int blocks = std::max(0, std::min({blocksThreads, blocksReg, blocksSmem}));
      return double(blocks * int(blockThreads)) / double(p.maxThreadsPerMultiProcessor);
    }

    void handleKernel(CUpti_ActivityKernel9 const* k) {
      char* demangled = abi::__cxa_demangle(k->name, nullptr, nullptr, nullptr);
      std::string const name = shorten(demangled ? demangled : k->name);
      std::string const full = demangled ? demangled : (k->name ? k->name : "");
      std::free(demangled);

      uint32_t const blockThreads = k->blockX * k->blockY * k->blockZ;
      uint32_t const smem = k->staticSharedMemory + k->dynamicSharedMemory;
      double const occ = occupancy(int(k->deviceId), k->registersPerThread, smem, blockThreads);
      std::string const grid =
          std::to_string(k->gridX) + "x" + std::to_string(k->gridY) + "x" + std::to_string(k->gridZ);
      std::string const block =
          std::to_string(k->blockX) + "x" + std::to_string(k->blockY) + "x" + std::to_string(k->blockZ);

      auto track = gpuTrack(k->deviceId, k->streamId);
      ::perfetto::TraceTimestamp const tsBegin{6, uint64_t(int64_t(k->start) + offsetNs_)};  // 6 == BOOTTIME
      ::perfetto::TraceTimestamp const tsEnd{6, uint64_t(int64_t(k->end) + offsetNs_)};

      TRACE_EVENT_BEGIN("cmssw.gpu",
                        ::perfetto::DynamicString{name.c_str()},
                        track,
                        tsBegin,
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
                        ::perfetto::DynamicString{grid.c_str()},
                        "block",
                        ::perfetto::DynamicString{block.c_str()},
                        "occupancy_est",
                        occ,
                        "correlation_id",
                        k->correlationId,
                        "kernel",
                        ::perfetto::DynamicString{full.c_str()});
      TRACE_EVENT_END("cmssw.gpu", track, tsEnd);
    }

    bool active_ = false;
    int64_t offsetNs_ = 0;
    std::vector<cudaDeviceProp> props_;
    std::mutex mutex_;
    std::set<uint64_t> seen_;

    static PerfettoCuptiProfiler* s_instance;
  };

  inline PerfettoCuptiProfiler* PerfettoCuptiProfiler::s_instance = nullptr;

}  // namespace cms::perfetto

#else  // PERFETTO_HAS_CUPTI

namespace cms::perfetto {

  // No CUDA/CUPTI on this architecture: a no-op stub so that the rest of the
  // service (lanes, counters, allocator, power) builds and runs normally.
  class PerfettoCuptiProfiler {
  public:
    bool start() { return false; }
    void flush() {}
    void stop() {}
  };

}  // namespace cms::perfetto

#endif  // PERFETTO_HAS_CUPTI

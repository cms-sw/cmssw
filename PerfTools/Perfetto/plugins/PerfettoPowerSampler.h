// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_plugins_PerfettoPowerSampler_h
#define PerfTools_Perfetto_plugins_PerfettoPowerSampler_h

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"

#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>

namespace cms::perfetto {

  // Samples power on a background thread into counter tracks:
  // "CPU pkg<n> power (W)" from the RAPL energy counters in /sys/class/powercap,
  // "GPU<d> power (W)" from NVML, loaded with dlopen (no build dependency).
  class PerfettoPowerSampler {
  public:
    ~PerfettoPowerSampler() { stop(); }

    // True if at least one power source was found and sampling started.
    bool start(std::chrono::milliseconds period) {
      openGpus();
      openCpus();
      if (gpus_.empty() && cpus_.empty()) {
        closeNvml();
        return false;
      }
      thread_ = std::jthread([this, period](std::stop_token token) { loop(token, period); });
      return true;
    }

    void stop() {
      if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
      }
      closeNvml();
    }

  private:
    using NvmlFn = int (*)();
    using NvmlCountFn = int (*)(unsigned*);
    using NvmlHandleFn = int (*)(unsigned, void**);
    using NvmlPowerFn = int (*)(void*, unsigned*);  // milliwatts

    struct Gpu {
      void* handle;
      ::perfetto::CounterTrack track;
    };

    struct Rapl {
      std::string path;
      uint64_t range;  // the counter wraps at this value (uJ)
      uint64_t lastEnergy;
      uint64_t lastTime;  // trace clock, ns
      ::perfetto::CounterTrack track;
    };

    static bool readU64(std::string const& path, uint64_t& value) {
      std::FILE* f = std::fopen(path.c_str(), "re");
      if (!f)
        return false;
      bool const ok = std::fscanf(f, "%" SCNu64, &value) == 1;
      std::fclose(f);
      return ok;
    }

    template <class Fn>
    Fn symbol(const char* name) const {
      return reinterpret_cast<Fn>(::dlsym(nvml_, name));
    }

    void openGpus() {
      nvml_ = ::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
      if (!nvml_)
        return;
      auto const init = symbol<NvmlFn>("nvmlInit_v2");
      auto const count = symbol<NvmlCountFn>("nvmlDeviceGetCount_v2");
      auto const handle = symbol<NvmlHandleFn>("nvmlDeviceGetHandleByIndex_v2");
      nvmlShutdown_ = symbol<NvmlFn>("nvmlShutdown");
      nvmlPower_ = symbol<NvmlPowerFn>("nvmlDeviceGetPowerUsage");
      if (!init || !count || !handle || !nvmlShutdown_ || !nvmlPower_ || init() != 0) {
        nvmlShutdown_ = nullptr;  // not initialized
        closeNvml();
        return;
      }
      unsigned n = 0;
      if (count(&n) != 0)
        return;
      for (unsigned i = 0; i < n; ++i) {
        void* h = nullptr;
        if (handle(i, &h) == 0)
          gpus_.push_back({h,
                           ::perfetto::CounterTrack(::perfetto::DynamicString(
                               names_.emplace_back("GPU" + std::to_string(i) + " power (W)")))});
      }
    }

    void openCpus() {
      for (int pkg = 0;; ++pkg) {
        std::string const base = "/sys/class/powercap/intel-rapl:" + std::to_string(pkg);
        uint64_t energy = 0;
        if (!readU64(base + "/energy_uj", energy))  // absent, or not readable by this user
          break;
        uint64_t range = 0;
        readU64(base + "/max_energy_range_uj", range);
        cpus_.push_back({base + "/energy_uj",
                         range ? range : ~uint64_t{0},
                         energy,
                         TrackEvent::GetTraceTimeNs(),
                         ::perfetto::CounterTrack(::perfetto::DynamicString(
                             names_.emplace_back("CPU pkg" + std::to_string(pkg) + " power (W)")))});
      }
    }

    void closeNvml() {
      if (!nvml_)
        return;
      if (nvmlShutdown_)
        nvmlShutdown_();
      ::dlclose(nvml_);
      nvml_ = nullptr;
      nvmlShutdown_ = nullptr;
      nvmlPower_ = nullptr;
    }

    void loop(std::stop_token const& token, std::chrono::milliseconds period) {
      std::mutex mutex;
      std::condition_variable_any cv;
      std::unique_lock lock(mutex);
      do {
        for (auto const& gpu : gpus_) {
          unsigned mW = 0;
          if (nvmlPower_(gpu.handle, &mW) == 0)
            TRACE_COUNTER("cmssw.power", gpu.track, mW / 1000.);
        }
        for (auto& cpu : cpus_) {
          uint64_t energy = 0;
          if (!readU64(cpu.path, energy))
            continue;
          uint64_t const now = TrackEvent::GetTraceTimeNs();
          uint64_t const dE = energy >= cpu.lastEnergy ? energy - cpu.lastEnergy : cpu.range - cpu.lastEnergy + energy;
          if (now > cpu.lastTime)
            TRACE_COUNTER("cmssw.power", cpu.track, double(dE) * 1e3 / double(now - cpu.lastTime));  // uJ/ns -> W
          cpu.lastEnergy = energy;
          cpu.lastTime = now;
        }
      } while (!cv.wait_for(lock, token, period, [&token] { return token.stop_requested(); }));
    }

    void* nvml_ = nullptr;
    NvmlFn nvmlShutdown_ = nullptr;
    NvmlPowerFn nvmlPower_ = nullptr;
    std::deque<std::string> names_;  // backs the counter track names: stable addresses
    std::vector<Gpu> gpus_;
    std::vector<Rapl> cpus_;
    std::jthread thread_;
  };

}  // namespace cms::perfetto

#endif  // PerfTools_Perfetto_plugins_PerfettoPowerSampler_h

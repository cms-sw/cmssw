// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#pragma once

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include <perfetto.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <time.h>

// Samples CPU and GPU power on a background thread and emits them as Perfetto
// counter tracks: "CPU pkg<n> power (W)" from the Intel RAPL energy counters
// (/sys/class/powercap/intel-rapl:*), and "GPU<d> power (W)" from NVML
// (nvmlDeviceGetPowerUsage). NVML is loaded with dlopen so there is no build-time
// dependency on it; if neither source is available the sampler is a no-op.
namespace cms::perfetto {

  class PerfettoPowerSampler {
  public:
    // Returns true if at least one power source (RAPL or NVML) was found and the
    // sampling thread was started. periodMs is kept modest by default because NVML
    // power queries are not free and can perturb the GPU if polled too often.
    bool start(unsigned periodMs = 1000) {
      periodMs_ = periodMs;
      openGpu();
      openCpu();
      if (gpus_.empty() && cpus_.empty())
        return false;
      running_ = true;
      thread_ = std::thread([this] { loop(); });
      return true;
    }

    void stop() {
      if (running_.exchange(false)) {
        cv_.notify_all();
        if (thread_.joinable())
          thread_.join();
      }
      if (nvml_) {
        if (nvmlShutdown_)
          nvmlShutdown_();
        ::dlclose(nvml_);
        nvml_ = nullptr;
      }
    }

    ~PerfettoPowerSampler() { stop(); }

  private:
    using FnInt = int (*)();
    using FnCount = int (*)(unsigned*);
    using FnHandle = int (*)(unsigned, void**);
    using FnPower = int (*)(void*, unsigned*);

    static int64_t nowNs() {
      timespec ts{};
      clock_gettime(CLOCK_BOOTTIME, &ts);
      return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    }

    static bool readU64(std::string const& path, uint64_t& out) {
      std::FILE* f = std::fopen(path.c_str(), "re");
      if (!f)
        return false;
      bool const ok = std::fscanf(f, "%lu", &out) == 1;
      std::fclose(f);
      return ok;
    }

    void openGpu() {
      nvml_ = ::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
      if (!nvml_)
        return;
      nvmlShutdown_ = reinterpret_cast<FnInt>(::dlsym(nvml_, "nvmlShutdown"));
      auto init = reinterpret_cast<FnInt>(::dlsym(nvml_, "nvmlInit_v2"));
      auto count = reinterpret_cast<FnCount>(::dlsym(nvml_, "nvmlDeviceGetCount_v2"));
      auto handle = reinterpret_cast<FnHandle>(::dlsym(nvml_, "nvmlDeviceGetHandleByIndex_v2"));
      nvmlPower_ = reinterpret_cast<FnPower>(::dlsym(nvml_, "nvmlDeviceGetPowerUsage"));
      if (!init || !count || !handle || !nvmlPower_ || init() != 0)
        return;
      unsigned n = 0;
      if (count(&n) != 0)
        return;
      for (unsigned i = 0; i < n; ++i) {
        void* h = nullptr;
        if (handle(i, &h) == 0) {
          gpus_.push_back(h);
          gpuNames_.push_back("GPU" + std::to_string(i) + " power (W)");
        }
      }
    }

    void openCpu() {
      for (int pkg = 0;; ++pkg) {
        std::string const base = "/sys/class/powercap/intel-rapl:" + std::to_string(pkg);
        uint64_t energy = 0;
        if (!readU64(base + "/energy_uj", energy))
          break;
        uint64_t range = 0;
        readU64(base + "/max_energy_range_uj", range);
        cpus_.push_back({base + "/energy_uj", energy, range ? range : ~uint64_t{0}, nowNs()});
        cpuNames_.push_back("CPU pkg" + std::to_string(pkg) + " power (W)");
      }
    }

    void loop() {
      while (running_.load()) {
        for (std::size_t i = 0; i < gpus_.size(); ++i) {
          unsigned mw = 0;
          if (nvmlPower_(gpus_[i], &mw) == 0)
            TRACE_COUNTER(
                "cmssw.power", ::perfetto::CounterTrack(::perfetto::DynamicString{gpuNames_[i].c_str()}), mw / 1000.0);
        }
        for (std::size_t i = 0; i < cpus_.size(); ++i) {
          uint64_t energy = 0;
          if (!readU64(cpus_[i].path, energy))
            continue;
          int64_t const now = nowNs();
          uint64_t const dE = (energy >= cpus_[i].lastEnergy) ? (energy - cpus_[i].lastEnergy)
                                                              : (cpus_[i].range - cpus_[i].lastEnergy + energy);
          double const dt = double(now - cpus_[i].lastTime) / 1e9;
          if (dt > 0.)
            TRACE_COUNTER("cmssw.power",
                          ::perfetto::CounterTrack(::perfetto::DynamicString{cpuNames_[i].c_str()}),
                          double(dE) / 1e6 / dt);
          cpus_[i].lastEnergy = energy;
          cpus_[i].lastTime = now;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(periodMs_), [this] { return !running_.load(); });
      }
    }

    struct Rapl {
      std::string path;
      uint64_t lastEnergy;
      uint64_t range;
      int64_t lastTime;
    };

    unsigned periodMs_ = 50;
    void* nvml_ = nullptr;
    FnInt nvmlShutdown_ = nullptr;
    FnPower nvmlPower_ = nullptr;
    std::vector<void*> gpus_;
    std::vector<Rapl> cpus_;
    std::vector<std::string> gpuNames_;
    std::vector<std::string> cpuNames_;

    std::atomic<bool> running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
  };

}  // namespace cms::perfetto

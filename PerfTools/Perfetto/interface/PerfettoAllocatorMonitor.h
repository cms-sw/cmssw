// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_interface_PerfettoAllocatorMonitor_h
#define PerfTools_Perfetto_interface_PerfettoAllocatorMonitor_h

#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocatorMonitor.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cms::perfetto {

  // Records caching-allocator transactions: each alloc/free is an instant on the
  // lane of the module that made it (so it sits under that module's slice), and
  // the live / cached / requested byte totals are per-memory-space counters.
  class PerfettoAllocatorMonitor final : public cms::alpakatools::CachingAllocatorMonitor {
  public:
    PerfettoAllocatorMonitor() {
      static constexpr std::array<const char*, kNumCounters> kSuffix{{" live (B)", " cached (B)", " requested (B)"}};
      for (int s = 0; s < kNumSpaces; ++s) {
        std::string const tag = (s == kHost) ? "host" : "dev" + std::to_string(s);
        for (int c = 0; c < kNumCounters; ++c)
          names_[s][c] = tag + kSuffix[c];
      }
      // after names_ is filled: the tracks keep pointers to its strings
      for (auto const& space : names_)
        for (auto const& name : space)
          counters_.emplace_back(::perfetto::DynamicString(name));
    }

    PerfettoAllocatorMonitor(PerfettoAllocatorMonitor const&) = delete;
    PerfettoAllocatorMonitor& operator=(PerfettoAllocatorMonitor const&) = delete;

    void onAllocate(Device device,
                    void const*,
                    std::size_t bytes,
                    std::size_t requested,
                    bool cacheHit,
                    unsigned long long queue) noexcept override {
      if (!TRACE_EVENT_CATEGORY_ENABLED("cmssw.alloc"))
        return;
      auto const& m = currentModuleContext();
      TRACE_EVENT_INSTANT("cmssw.alloc",
                          "alloc",
                          trackFor(m),
                          "module",
                          ::perfetto::DynamicString(m.label ? m.label : "(none)"),
                          "bytes",
                          static_cast<uint64_t>(bytes),
                          "requested",
                          static_cast<uint64_t>(requested),
                          "cache_hit",
                          cacheHit,
                          "device",
                          device.host ? -1 : device.index,
                          "queue",
                          queue);
    }

    void onFree(Device device, void const*, std::size_t bytes, unsigned long long queue) noexcept override {
      if (!TRACE_EVENT_CATEGORY_ENABLED("cmssw.alloc"))
        return;
      auto const& m = currentModuleContext();
      TRACE_EVENT_INSTANT("cmssw.alloc",
                          "free",
                          trackFor(m),
                          "module",
                          ::perfetto::DynamicString(m.label ? m.label : "(none)"),
                          "bytes",
                          static_cast<uint64_t>(bytes),
                          "device",
                          device.host ? -1 : device.index,
                          "queue",
                          queue);
    }

    void onUsage(Device device, std::size_t live, std::size_t cached, std::size_t requested) noexcept override {
      int const space = device.host ? kHost : device.index;
      if (space < 0 || space >= kNumSpaces || !TRACE_EVENT_CATEGORY_ENABLED("cmssw.alloc"))
        return;
      auto const* track = &counters_[space * kNumCounters];
      TRACE_COUNTER("cmssw.alloc", track[0], live);
      TRACE_COUNTER("cmssw.alloc", track[1], cached);
      TRACE_COUNTER("cmssw.alloc", track[2], requested);
    }

  private:
    static ::perfetto::Track trackFor(ModuleContext const& m) {
      if (m.label && m.streamId != ModuleContext::kNoStream)
        return laneTrack(m.streamId);
      return ::perfetto::ThreadTrack::Current();
    }

    static constexpr int kMaxDevices = 16;  // devices beyond this get no counters
    static constexpr int kHost = kMaxDevices;
    static constexpr int kNumSpaces = kMaxDevices + 1;
    static constexpr int kNumCounters = 3;  // live, cached, requested

    std::array<std::array<std::string, kNumCounters>, kNumSpaces> names_;
    std::vector<::perfetto::CounterTrack> counters_;  // [space * kNumCounters + counter]
  };

}  // namespace cms::perfetto

#endif  // PerfTools_Perfetto_interface_PerfettoAllocatorMonitor_h

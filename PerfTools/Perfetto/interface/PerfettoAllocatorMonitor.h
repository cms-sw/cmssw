// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#pragma once

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
#include <perfetto.h>

#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocatorMonitor.h"

#include <array>
#include <cstdint>
#include <string>

namespace cms::perfetto {

  // Turns Alpaka caching-allocator transactions into Perfetto events. Each
  // alloc/free is emitted as an INSTANT on the calling thread's track -- so it
  // sits, visually and by annotation, under the module slice that triggered it --
  // and the live / cached / requested byte totals are emitted as per-device
  // counters, giving a device-memory-pressure timeline. Attribution to the
  // responsible module comes from the thread-local cms::perfetto::ModuleContext.
  class PerfettoAllocatorMonitor : public cms::alpakatools::CachingAllocatorMonitor {
  public:
    PerfettoAllocatorMonitor() {
      for (int d = 0; d <= kMaxDevice; ++d) {
        std::string const tag = (d == kMaxDevice) ? std::string("host") : ("dev" + std::to_string(d));
        names_[d][0] = tag + " live (B)";
        names_[d][1] = tag + " cached (B)";
        names_[d][2] = tag + " requested (B)";
      }
    }

    void onAllocate(int device,
                    const void* /*ptr*/,
                    std::size_t bytes,
                    std::size_t requested,
                    bool cacheHit,
                    unsigned long long queue) noexcept override {
      if (!::perfetto::TrackEvent::IsEnabled())
        return;
      auto const& m = cms::perfetto::currentModuleContext();
      TRACE_EVENT_INSTANT("cmssw.alloc",
                          "alloc",
                          trackFor(m),
                          "module",
                          ::perfetto::DynamicString(m.active && m.label ? m.label : "(none)"),
                          "bytes",
                          static_cast<uint64_t>(bytes),
                          "requested",
                          static_cast<uint64_t>(requested),
                          "cache_hit",
                          cacheHit,
                          "device",
                          device,
                          "queue",
                          queue);
    }

    void onFree(int device, const void* /*ptr*/, std::size_t bytes, unsigned long long queue) noexcept override {
      if (!::perfetto::TrackEvent::IsEnabled())
        return;
      auto const& m = cms::perfetto::currentModuleContext();
      TRACE_EVENT_INSTANT("cmssw.alloc",
                          "free",
                          trackFor(m),
                          "module",
                          ::perfetto::DynamicString(m.active && m.label ? m.label : "(none)"),
                          "bytes",
                          static_cast<uint64_t>(bytes),
                          "device",
                          device,
                          "queue",
                          queue);
    }

    void onUsage(int device, std::size_t live, std::size_t cached, std::size_t requested) noexcept override {
      if (!::perfetto::TrackEvent::IsEnabled())
        return;
      int const d = (device < 0 || device >= kMaxDevice) ? kMaxDevice : device;
      TRACE_COUNTER("cmssw.alloc", ::perfetto::CounterTrack(::perfetto::DynamicString{names_[d][0].c_str()}), live);
      TRACE_COUNTER("cmssw.alloc", ::perfetto::CounterTrack(::perfetto::DynamicString{names_[d][1].c_str()}), cached);
      TRACE_COUNTER(
          "cmssw.alloc", ::perfetto::CounterTrack(::perfetto::DynamicString{names_[d][2].c_str()}), requested);
    }

  private:
    // The alloc/free instant goes on the lane of the module that triggered it (so it
    // sits under that module's slice). Outside any traced module it falls back to the
    // calling thread's own track.
    static ::perfetto::Track trackFor(cms::perfetto::ModuleContext const& m) {
      if (m.active && m.streamId != 0xffffffffu)
        return cms::perfetto::laneTrack(m.streamId);
      return ::perfetto::ThreadTrack::Current();
    }

    static constexpr int kMaxDevice = 16;  // index kMaxDevice == host
    std::array<std::array<std::string, 3>, kMaxDevice + 1> names_;
  };

}  // namespace cms::perfetto

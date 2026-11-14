// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#include <atomic>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocatorMonitor.h"

#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
#include "PerfTools/Perfetto/interface/PerfettoAllocatorMonitor.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  constexpr size_t SIZE = 1024;

  // Counts transactions and records the module they are attributed to.
  struct MockMonitor : public cms::alpakatools::CachingAllocatorMonitor {
    std::atomic<int> allocs{0};
    std::atomic<int> frees{0};
    std::string allocModule;
    std::string freeModule;

    void onAllocate(Device, void const*, std::size_t, std::size_t, bool, unsigned long long) noexcept override {
      ++allocs;
      if (auto const* label = cms::perfetto::currentModuleContext().label)
        allocModule = label;
    }
    void onFree(Device, void const*, std::size_t, unsigned long long) noexcept override {
      ++frees;
      if (auto const* label = cms::perfetto::currentModuleContext().label)
        freeModule = label;
    }
  };
}  // namespace

TEST_CASE("Caching-allocator monitor hook attributes transactions to the current module (" EDM_STRINGIZE(
              ALPAKA_ACCELERATOR_NAMESPACE) ")",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    INFO("no devices available for this backend, skipping");
    return;
  }

  SECTION("mock monitor sees alloc/free with the module attribution") {
    MockMonitor mon;
    cms::alpakatools::setCachingAllocatorMonitor(&mon);

    {
      // as if inside module "testAllocModule" on stream 0
      cms::perfetto::ModuleContextGuard guard({"testAllocModule", 0});
      auto queue = Queue(devices[0]);
      auto buf_h = cms::alpakatools::make_host_buffer<float[]>(queue, SIZE);
      auto buf_d = cms::alpakatools::make_device_buffer<float[]>(queue, SIZE);
      alpaka::memset(queue, buf_d, 0);
      alpaka::wait(queue);
    }  // buffers freed here -> onFree
    cms::alpakatools::setCachingAllocatorMonitor(nullptr);

    // Not every backend routes buffers through the caching allocator (e.g. the
    // serial-CPU backend allocates directly). Where it does, the monitor must
    // have been called and the transactions attributed to the current module.
    if (mon.allocs.load() == 0) {
      WARN("this backend does not use the caching allocator; monitor not exercised");
    } else {
      REQUIRE(mon.frees.load() >= 1);
      REQUIRE(mon.allocModule == "testAllocModule");
      REQUIRE(mon.freeModule == "testAllocModule");
    }
  }

  SECTION("PerfettoAllocatorMonitor records the attributed transactions and device counters") {
    ::perfetto::TracingInitArgs args;
    args.backends = ::perfetto::kInProcessBackend;
    ::perfetto::Tracing::Initialize(args);
    cms::perfetto::TrackEvent::Register();

    ::perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(4096);
    auto* ds = cfg.add_data_sources()->mutable_config();
    ds->set_name("track_event");
    ::perfetto::protos::gen::TrackEventConfig te;
    te.add_enabled_categories("cmssw.alloc");
    ds->set_track_event_config_raw(te.SerializeAsString());

    auto session = ::perfetto::Tracing::NewTrace();
    session->Setup(cfg);
    session->StartBlocking();

    cms::perfetto::PerfettoAllocatorMonitor monitor;
    cms::alpakatools::setCachingAllocatorMonitor(&monitor);
    {
      cms::perfetto::ModuleContextGuard guard({"testAllocModule", 0});
      auto queue = Queue(devices[0]);
      auto buf_d = cms::alpakatools::make_device_buffer<float[]>(queue, SIZE);
      alpaka::memset(queue, buf_d, 0);
      alpaka::wait(queue);
    }
    cms::alpakatools::setCachingAllocatorMonitor(nullptr);

    cms::perfetto::TrackEvent::Flush();
    session->StopBlocking();
    std::vector<char> const data = session->ReadTraceBlocking();
    REQUIRE(not data.empty());
    // device buffers go through the caching allocator on the GPU backends only
    if constexpr (not std::is_same_v<Device, alpaka::DevCpu>) {
      std::string_view const trace(data.data(), data.size());
      REQUIRE(trace.find("testAllocModule") != std::string_view::npos);
      REQUIRE(trace.find("live (B)") != std::string_view::npos);
    }
  }
}

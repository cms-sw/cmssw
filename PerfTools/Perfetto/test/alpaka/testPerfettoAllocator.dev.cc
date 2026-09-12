// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#include <atomic>
#include <string>

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
#include <perfetto.h>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  constexpr size_t SIZE = 1024;

  // Counts transactions and captures the module attribution seen via the
  // thread-local cms::perfetto::ModuleContext.
  struct MockMonitor : public cms::alpakatools::CachingAllocatorMonitor {
    std::atomic<int> allocs{0};
    std::atomic<int> frees{0};
    std::string allocModule;
    std::string freeModule;

    void onAllocate(int, const void*, std::size_t, std::size_t, bool, unsigned long long) noexcept override {
      ++allocs;
      auto const& m = cms::perfetto::currentModuleContext();
      if (m.active && m.label)
        allocModule = m.label;
    }
    void onFree(int, const void*, std::size_t, unsigned long long) noexcept override {
      ++frees;
      auto const& m = cms::perfetto::currentModuleContext();
      if (m.active && m.label)
        freeModule = m.label;
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

    // pretend we are inside module "testAllocModule" running on stream 0
    cms::perfetto::ModuleContext ctx;
    ctx.label = "testAllocModule";
    ctx.streamId = 0;
    ctx.active = true;
    cms::perfetto::pushModuleContext(ctx);

    {
      auto queue = Queue(devices[0]);
      auto buf_h = cms::alpakatools::make_host_buffer<float[]>(queue, SIZE);
      auto buf_d = cms::alpakatools::make_device_buffer<float[]>(queue, SIZE);
      alpaka::memset(queue, buf_d, 0);
      alpaka::wait(queue);
    }  // buffers freed here -> onFree

    cms::perfetto::popModuleContext();
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

  SECTION("PerfettoAllocatorMonitor emits a non-empty trace") {
    ::perfetto::TracingInitArgs args;
    args.backends = ::perfetto::kInProcessBackend;
    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();

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
    cms::perfetto::ModuleContext ctx;
    ctx.label = "testAllocModule";
    ctx.active = true;
    cms::perfetto::pushModuleContext(ctx);

    {
      auto queue = Queue(devices[0]);
      auto buf_d = cms::alpakatools::make_device_buffer<float[]>(queue, SIZE);
      alpaka::memset(queue, buf_d, 0);
      alpaka::wait(queue);
    }

    cms::perfetto::popModuleContext();
    cms::alpakatools::setCachingAllocatorMonitor(nullptr);

    ::perfetto::TrackEvent::Flush();
    session->StopBlocking();
    std::vector<char> data = session->ReadTraceBlocking();
    REQUIRE(not data.empty());
  }
}

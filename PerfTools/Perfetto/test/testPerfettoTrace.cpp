// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
//
// Records small traces the way PerfettoTraceService does and decodes them with
// perfetto's own protozero decoders, so that an SDK update or a change to the
// helpers that breaks track parenting, counters or slice balance fails loudly.

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoTrace.h"

namespace {
  // Perfetto may be initialized only once per process.
  void initPerfettoOnce() {
    [[maybe_unused]] static const bool done = [] {
      ::perfetto::TracingInitArgs args;
      args.backends = ::perfetto::kInProcessBackend;
      ::perfetto::Tracing::Initialize(args);
      cms::perfetto::TrackEvent::Register();
      return true;
    }();
  }

  // Run |emit| in a fresh in-process session with only |categories| enabled, as
  // the service configures it, and return the serialized trace.
  std::vector<char> recordTrace(std::initializer_list<const char*> categories, const std::function<void()>& emit) {
    initPerfettoOnce();
    ::perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(8192);
    auto* ds = cfg.add_data_sources()->mutable_config();
    ds->set_name("track_event");
    ::perfetto::protos::gen::TrackEventConfig te;
    te.add_disabled_categories("*");
    for (const char* c : categories)
      te.add_enabled_categories(c);
    ds->set_track_event_config_raw(te.SerializeAsString());

    auto session = ::perfetto::Tracing::NewTrace();
    session->Setup(cfg);
    session->StartBlocking();
    emit();
    cms::perfetto::TrackEvent::Flush();
    session->StopBlocking();
    return session->ReadTraceBlocking();
  }

  bool contains(std::vector<char> const& bytes, std::string_view what) {
    return std::string_view(bytes.data(), bytes.size()).find(what) != std::string_view::npos;
  }
}  // namespace

TEST_CASE("Perfetto trace has the expected track/counter/slice structure", "[perfetto]") {
  // no using-directive for cms::perfetto here: it would make the TRACE_* macros ambiguous
  using cms::perfetto::laneTrack;
  using cms::perfetto::streamTrack;
  auto bytes = recordTrace({"cmssw.event", "cmssw.module"}, [] {
    auto proc = ::perfetto::ProcessTrack::Current();
    {
      auto d = proc.Serialize();
      d.mutable_process()->set_process_name("cmsRun");
      cms::perfetto::TrackEvent::SetTrackDescriptor(proc, d);
    }
    for (unsigned s = 0; s < 2; ++s) {
      auto st = streamTrack(s);
      {
        auto d = st.Serialize();
        d.set_name("edm::stream " + std::to_string(s));
        cms::perfetto::TrackEvent::SetTrackDescriptor(st, d);
      }
      TRACE_EVENT_BEGIN("cmssw.event", "Event", st, "run", 1, "lumi", 1, "event", static_cast<int>(s + 1));
      TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("run", "id", st), 1.0);
      auto lane = laneTrack(s);
      TRACE_EVENT_BEGIN("cmssw.module", "outer", lane);
      TRACE_EVENT_BEGIN("cmssw.module", "inner", lane);  // nests inside outer on the same lane
      TRACE_EVENT_END("cmssw.module", lane);
      TRACE_EVENT_END("cmssw.module", lane);
      TRACE_EVENT_END("cmssw.event", st);
      TRACE_COUNTER("cmssw.event", ::perfetto::CounterTrack("Throughput (events/s)"), 100.0 + s);
    }
  });

  REQUIRE(!bytes.empty());

  namespace pb = ::perfetto::protos::pbzero;
  pb::Trace::Decoder trace(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());

  // the process-wide Throughput counter is checked by the presence of its name
  bool const sawThroughput = contains(bytes, "Throughput (events/s)");

  bool sawProcess = false;
  int laneTracks = 0, packets = 0;
  int begins = 0, ends = 0, counters = 0;
  std::set<uint64_t> streamUuids;

  // First pass: process/stream/counter descriptors and slice/counter events.
  for (auto it = trace.packet(); it; ++it) {
    ++packets;
    pb::TracePacket::Decoder packet(*it);
    if (packet.has_track_descriptor()) {
      pb::TrackDescriptor::Decoder td(packet.track_descriptor());
      std::string const name = td.name().ToStdString();
      if (td.has_process()) {
        pb::ProcessDescriptor::Decoder p(td.process());
        if (p.process_name().ToStdString() == "cmsRun")
          sawProcess = true;
      }
      if (name.rfind("edm::stream", 0) == 0)
        streamUuids.insert(td.uuid());
    }
    if (packet.has_track_event()) {
      pb::TrackEvent::Decoder te(packet.track_event());
      switch (te.type()) {
        case pb::TrackEvent::TYPE_SLICE_BEGIN:
          ++begins;
          break;
        case pb::TrackEvent::TYPE_SLICE_END:
          ++ends;
          break;
        case pb::TrackEvent::TYPE_COUNTER:
          ++counters;
          break;
        default:
          break;
      }
    }
  }

  // Second pass: lanes are descriptors parented to a stream track -- exactly the
  // property the per-(stream,thread) lane scheme must preserve.
  pb::Trace::Decoder trace2(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  for (auto it = trace2.packet(); it; ++it) {
    pb::TracePacket::Decoder packet(*it);
    if (packet.has_track_descriptor()) {
      pb::TrackDescriptor::Decoder td(packet.track_descriptor());
      if (streamUuids.count(td.parent_uuid()))
        ++laneTracks;
    }
  }

  REQUIRE(packets > 0);
  REQUIRE(sawProcess);
  REQUIRE(streamUuids.size() == 2u);
  REQUIRE(laneTracks >= 1);
  REQUIRE(sawThroughput);
  REQUIRE(begins == ends);  // every slice is closed
  REQUIRE(begins >= 6);     // 2 streams x (Event + outer + inner)
  REQUIRE(counters >= 4);   // 2 streams x (run counter + Throughput counter)
}

TEST_CASE("lane and stream track uuids are distinct and correctly parented", "[perfetto]") {
  using namespace cms::perfetto;
  initPerfettoOnce();

  auto s0 = streamTrack(0);
  auto s1 = streamTrack(1);
  REQUIRE(s0.uuid != 0);
  REQUIRE(s0.uuid != s1.uuid);  // one track per stream

  auto l0 = laneTrack(0);
  auto l1 = laneTrack(1);
  REQUIRE(l0.uuid != 0);                  // a lane must not collapse onto the root (the old xor-cancellation bug)
  REQUIRE(l0.uuid != s0.uuid);            // lane is its own track, not the stream track
  REQUIRE(l0.uuid != l1.uuid);            // different streams -> different lanes
  REQUIRE(l0.parent_uuid == s0.uuid);     // the lane hangs under its stream
  REQUIRE(laneTrack(0).uuid == l0.uuid);  // stable for the same (stream, thread)
}

TEST_CASE("CMS_PERFETTO_SCOPE is recorded only when cmssw.func is enabled", "[perfetto]") {
  auto const emit = [] {
    cms::perfetto::ModuleContextGuard guard({"someModule", 0});
    CMS_PERFETTO_SCOPE("scopeUnderTest");
  };
  REQUIRE(contains(recordTrace({"cmssw.event", "cmssw.func"}, emit), "scopeUnderTest"));
  REQUIRE_FALSE(contains(recordTrace({"cmssw.event"}, emit), "scopeUnderTest"));
}

TEST_CASE("module context stack nests and propagates across threads", "[perfetto]") {
  using namespace cms::perfetto;
  REQUIRE(currentModuleContext().label == nullptr);

  REQUIRE(pushModuleContext({"A", 3}));
  REQUIRE(std::string(currentModuleContext().label) == "A");
  REQUIRE(currentModuleContext().streamId == 3u);

  {
    ModuleContextGuard guard({"B", 3});
    REQUIRE(std::string(currentModuleContext().label) == "B");
    ModuleContextGuard untraced({});  // an untraced module nested in B hides it
    REQUIRE(currentModuleContext().label == nullptr);
  }
  REQUIRE(std::string(currentModuleContext().label) == "A");

  // withModuleContext carries the context onto another thread
  std::string seenOnThread;
  auto body = withModuleContext([&seenOnThread] {
    auto const* label = currentModuleContext().label;
    seenOnThread = label ? label : "";
  });
  std::thread th([&body] { body(); });
  th.join();
  REQUIRE(seenOnThread == "A");

  REQUIRE(std::string(popModuleContext().label) == "A");
  REQUIRE(currentModuleContext().label == nullptr);
  REQUIRE(popModuleContext().label == nullptr);  // popping an empty stack is harmless
}

TEST_CASE("module context stack stays balanced past its capacity", "[perfetto]") {
  using namespace cms::perfetto;
  constexpr int kDeep = 100;  // more than the stack holds
  int recorded = 0;
  for (int i = 0; i < kDeep; ++i)
    recorded += pushModuleContext({"deep", 0});
  REQUIRE(recorded > 0);
  REQUIRE(recorded < kDeep);
  int traced = 0;
  for (int i = 0; i < kDeep; ++i)
    traced += popModuleContext().label != nullptr;
  REQUIRE(traced == recorded);  // every recorded push, and only those, pops a traced entry
  REQUIRE(currentModuleContext().label == nullptr);
}

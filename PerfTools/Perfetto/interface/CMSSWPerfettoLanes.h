// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#pragma once

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include <perfetto.h>

#include <atomic>
#include <cstdint>
#include <set>
#include <string>

// Per-stream "lane" tracks.
//
// CMSSW serializes events within a stream but runs the modules of one event as
// concurrent TBB tasks on different threads (and an ExternalWork module's
// acquire()/produce() run on different threads). To show that work *under* its
// edm::stream -- instead of scattered across process-wide thread tracks -- each
// slice is emitted on a lane keyed by (stream, executing thread), parented to the
// stream track. Because every lane is therefore fed by exactly one thread, the
// begin/end events on it arrive in order and nest correctly; concurrent work in
// the same stream simply lands on separate lanes shown side by side.
namespace cms::perfetto {

  inline constexpr uint64_t kStreamBase = 0x5354524D00000000ull;  // "STRM...."
  inline constexpr uint64_t kLaneBase = 0x4C414E4500000000ull;    // "LANE...."

  inline uint64_t streamUuid(unsigned sid) noexcept { return kStreamBase | (uint64_t{sid} << 16); }

  inline ::perfetto::Track streamTrack(unsigned sid) {
    return ::perfetto::Track(streamUuid(sid), ::perfetto::ProcessTrack::Current());
  }

  // A small, stable ordinal for the calling OS/TBB thread (assigned on first use).
  inline unsigned threadOrdinal() noexcept {
    static std::atomic<unsigned> next{0};
    static thread_local unsigned ord = next.fetch_add(1, std::memory_order_relaxed);
    return ord;
  }

  // The lane for the current thread within stream |sid|, as a child of the stream
  // track. The lane uuid embeds the thread ordinal, so it is owned by a single
  // thread -- the descriptor can be written from a thread-local guard, no lock.
  inline ::perfetto::Track laneTrack(unsigned sid) {
    unsigned const ord = threadOrdinal();
    uint64_t const uuid = kLaneBase | (uint64_t{sid} << 24) | ord;
    static thread_local std::set<unsigned> described;  // streams this thread has named a lane for
    if (described.insert(sid).second) {
      ::perfetto::Track t(uuid, streamTrack(sid));
      auto d = t.Serialize();
      d.set_name("thread " + std::to_string(ord));
      ::perfetto::TrackEvent::SetTrackDescriptor(t, d);
    }
    return ::perfetto::Track(uuid, streamTrack(sid));
  }

}  // namespace cms::perfetto

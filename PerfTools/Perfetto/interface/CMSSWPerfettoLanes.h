// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_interface_CMSSWPerfettoLanes_h
#define PerfTools_Perfetto_interface_CMSSWPerfettoLanes_h

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

// Per-stream tracks and per-(stream, thread) lanes.
//
// Modules of one event run concurrently on different threads, so their slices
// cannot share a track. Each slice goes on the lane of (its stream, the executing
// thread), a child of the stream track: a lane is fed by a single thread, so its
// slices are ordered and nest correctly.
//
// perfetto derives a child uuid as id ^ parent.uuid, so ids must not re-encode
// the parent's id: it would cancel out and merge the lanes of different streams.
namespace cms::perfetto {

  inline constexpr uint64_t kStreamBase = 0x5354524D00000000ull;  // "STRM"
  inline constexpr uint64_t kLaneBase = 0x4C414E4500000000ull;    // "LANE"

  inline ::perfetto::Track streamTrack(unsigned sid) {
    return ::perfetto::Track(kStreamBase | (uint64_t{sid} << 16), ::perfetto::ProcessTrack::Current());
  }

  // Small, stable ordinal of the calling thread, assigned on first use.
  inline unsigned threadOrdinal() noexcept {
    static std::atomic<unsigned> next{0};
    static thread_local unsigned const ord = next.fetch_add(1, std::memory_order_relaxed);
    return ord;
  }

  // The calling thread's lane in stream |sid|. Only this thread uses the lane, so
  // it names it on first use without locking.
  inline ::perfetto::Track laneTrack(unsigned sid) {
    unsigned const ord = threadOrdinal();
    ::perfetto::Track const lane(kLaneBase | ord, streamTrack(sid));
    static thread_local std::vector<bool> named;  // indexed by stream
    if (sid >= named.size())
      named.resize(sid + 1, false);
    if (!named[sid]) {
      named[sid] = true;
      auto desc = lane.Serialize();
      desc.set_name("thread " + std::to_string(ord));
      TrackEvent::SetTrackDescriptor(lane, desc);
    }
    return lane;
  }

}  // namespace cms::perfetto

#endif  // PerfTools_Perfetto_interface_CMSSWPerfettoLanes_h

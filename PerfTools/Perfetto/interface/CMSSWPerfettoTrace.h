// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_interface_CMSSWPerfettoTrace_h
#define PerfTools_Perfetto_interface_CMSSWPerfettoTrace_h

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"
#include <perfetto.h>

#include <optional>

// Tier-B, per-function instrumentation. The slice is emitted on the lane of the
// module currently executing on this thread (from cms::perfetto::ModuleContext),
// so it nests under that module's slice; outside any module it falls back to the
// calling thread's own track. Because a SliceScope lives entirely on one thread,
// its captured begin/end track always match.
namespace cms::perfetto_trace {

  struct SliceScope {
    explicit SliceScope(const char* name) noexcept {
      if (!::perfetto::TrackEvent::IsEnabled())
        return;
      auto const& m = cms::perfetto::currentModuleContext();
      if (m.active && m.streamId != 0xffffffffu) {
        track_.emplace(cms::perfetto::laneTrack(m.streamId));
        TRACE_EVENT_BEGIN("cmssw.func", ::perfetto::DynamicString(name), *track_);
      } else {
        TRACE_EVENT_BEGIN("cmssw.func", ::perfetto::DynamicString(name));
      }
      active_ = true;
    }

    ~SliceScope() noexcept {
      if (!active_)
        return;
      if (track_)
        TRACE_EVENT_END("cmssw.func", *track_);
      else
        TRACE_EVENT_END("cmssw.func");
    }

    SliceScope(SliceScope const&) = delete;
    SliceScope& operator=(SliceScope const&) = delete;

  private:
    std::optional<::perfetto::Track> track_;
    bool active_ = false;
  };

}  // namespace cms::perfetto_trace

#define CMS_PERFETTO_FUNC() \
  cms::perfetto_trace::SliceScope PERFETTO_UID(_cms_perfetto_func_) { __func__ }

#define CMS_PERFETTO_SCOPE(name_literal) \
  cms::perfetto_trace::SliceScope PERFETTO_UID(_cms_perfetto_scope_) { name_literal }

#endif  // PerfTools_Perfetto_interface_CMSSWPerfettoTrace_h

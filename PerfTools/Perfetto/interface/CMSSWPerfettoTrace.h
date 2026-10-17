// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_interface_CMSSWPerfettoTrace_h
#define PerfTools_Perfetto_interface_CMSSWPerfettoTrace_h

#include "PerfTools/Perfetto/interface/CMSSWPerfettoCategories.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoLanes.h"
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"

#include <optional>

namespace cms::perfetto {

  // Scoped slice for intra-module instrumentation, recorded only when the service
  // runs with traceFunctions=True. It nests under the slice of the module running
  // on this thread, or goes on the thread's own track outside any traced module.
  class SliceScope {
  public:
    explicit SliceScope(const char* name) noexcept {
      if (!TRACE_EVENT_CATEGORY_ENABLED("cmssw.func"))
        return;
      auto const& m = currentModuleContext();
      if (m.label && m.streamId != ModuleContext::kNoStream)
        track_.emplace(laneTrack(m.streamId));
      else
        track_.emplace(::perfetto::ThreadTrack::Current());
      TRACE_EVENT_BEGIN("cmssw.func", ::perfetto::DynamicString(name), *track_);
    }

    ~SliceScope() noexcept {
      if (track_)
        TRACE_EVENT_END("cmssw.func", *track_);
    }

    SliceScope(SliceScope const&) = delete;
    SliceScope& operator=(SliceScope const&) = delete;

  private:
    std::optional<::perfetto::Track> track_;
  };

}  // namespace cms::perfetto

#define CMS_PERFETTO_FUNC() \
  cms::perfetto::SliceScope PERFETTO_UID(cms_perfetto_func_) { __func__ }

#define CMS_PERFETTO_SCOPE(name) \
  cms::perfetto::SliceScope PERFETTO_UID(cms_perfetto_scope_) { name }

#endif  // PerfTools_Perfetto_interface_CMSSWPerfettoTrace_h

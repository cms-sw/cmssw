// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"

namespace cms::perfetto {
  namespace {
    // Allocation-free; nesting from work-stealing is shallow in practice. Past the
    // cap the depth is still counted, so push/pop stay balanced.
    constexpr unsigned kMaxDepth = 64;
    thread_local ModuleContext g_stack[kMaxDepth];
    thread_local unsigned g_depth = 0;
    constexpr ModuleContext g_none{};
  }  // namespace

  bool pushModuleContext(ModuleContext const& ctx) noexcept {
    bool const recorded = g_depth < kMaxDepth;
    if (recorded)
      g_stack[g_depth] = ctx;
    ++g_depth;
    return recorded;
  }

  ModuleContext popModuleContext() noexcept {
    if (g_depth == 0)
      return g_none;
    --g_depth;
    return g_depth < kMaxDepth ? g_stack[g_depth] : g_none;
  }

  ModuleContext const& currentModuleContext() noexcept {
    return (g_depth > 0 && g_depth <= kMaxDepth) ? g_stack[g_depth - 1] : g_none;
  }
}  // namespace cms::perfetto

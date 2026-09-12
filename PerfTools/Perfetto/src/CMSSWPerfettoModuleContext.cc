// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#include "PerfTools/Perfetto/interface/CMSSWPerfettoModuleContext.h"

namespace cms::perfetto {
  namespace {
    // A small fixed-size, allocation-free per-thread stack. Module re-entrancy
    // from work-stealing is shallow in practice; beyond the cap we keep counting
    // depth (so push/pop stay balanced) but stop recording, and report no module.
    constexpr int kMaxDepth = 64;
    thread_local ModuleContext g_stack[kMaxDepth];
    thread_local int g_depth = 0;
    const ModuleContext g_none{};
  }  // namespace

  void pushModuleContext(ModuleContext const& ctx) noexcept {
    if (g_depth >= 0 && g_depth < kMaxDepth)
      g_stack[g_depth] = ctx;
    ++g_depth;
  }

  void popModuleContext() noexcept {
    if (g_depth > 0)
      --g_depth;
  }

  void resetModuleContext() noexcept { g_depth = 0; }

  ModuleContext const& currentModuleContext() noexcept {
    if (g_depth > 0 && g_depth <= kMaxDepth)
      return g_stack[g_depth - 1];
    return g_none;
  }
}  // namespace cms::perfetto

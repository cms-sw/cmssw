// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#pragma once

#include <type_traits>
#include <utility>

namespace cms::perfetto {

  // Thread-local record of which CMSSW module (if any) is currently executing on
  // this thread. PerfettoTraceService pushes it around every module/acquire call,
  // so lower-level instrumentation that runs *inside* module code -- the
  // caching-allocator monitor and the GPU/CUPTI layer -- can attribute its work
  // to the responsible module without being passed any context.
  //
  // It is a *stack*, not a single slot, on purpose: TBB work-stealing can run
  // another module's produce() on this same thread while the current one is
  // blocked in a tbb::parallel_for (or similar), nested inside it. A stack
  // restores the enclosing module's context when that nested work returns.
  //
  // The char pointers reference the module's ModuleDescription, which outlives
  // the call, so they stay valid while the context is on the stack.
  struct ModuleContext {
    const char* label = nullptr;
    const char* type = nullptr;
    unsigned moduleId = 0;
    unsigned streamId = 0xffffffffu;
    unsigned long long eventId = 0;
    bool active = false;
  };

  void pushModuleContext(ModuleContext const& ctx) noexcept;
  void popModuleContext() noexcept;
  void resetModuleContext() noexcept;  // empty the stack (defensive, at boundaries)
  ModuleContext const& currentModuleContext() noexcept;

  // ---- Propagating the context across nested parallelism --------------------
  //
  // The context is thread-local and is pushed only on the thread that runs a
  // module's produce()/acquire(). If that code spawns work on *other* threads
  // (a tbb::parallel_for, edm::Async, ...), those helper threads do NOT see the
  // context, so allocations and tier-B slices they make are not attributed to
  // the module. Capture the context and re-apply it on the helper thread.

  // RAII: push a captured context for a scope, pop it on exit. Nests correctly.
  class ModuleContextGuard {
  public:
    explicit ModuleContextGuard(ModuleContext const& ctx) noexcept { pushModuleContext(ctx); }
    ~ModuleContextGuard() noexcept { popModuleContext(); }
    ModuleContextGuard(ModuleContextGuard const&) = delete;
    ModuleContextGuard& operator=(ModuleContextGuard const&) = delete;
  };

  // Wrap a callable so it runs with the *current* module context applied. Use it
  // as the body of a tbb::parallel_for/parallel_reduce (or anything that runs on
  // borrowed threads) so allocations and CMS_PERFETTO_FUNC slices made inside are
  // still attributed to the enclosing module:
  //
  //   tbb::parallel_for(range, cms::perfetto::withModuleContext([&](auto const& r){ ... }));
  template <class F>
  auto withModuleContext(F&& f) {
    return [ctx = currentModuleContext(), fn = std::decay_t<F>(std::forward<F>(f))](auto&&... args) mutable {
      ModuleContextGuard guard(ctx);
      return fn(std::forward<decltype(args)>(args)...);
    };
  }

}  // namespace cms::perfetto

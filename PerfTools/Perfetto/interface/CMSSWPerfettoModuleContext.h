// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_interface_CMSSWPerfettoModuleContext_h
#define PerfTools_Perfetto_interface_CMSSWPerfettoModuleContext_h

#include <type_traits>
#include <utility>

namespace cms::perfetto {

  // The CMSSW module running on this thread, published by PerfettoTraceService
  // around every module call so that the allocator monitor and CMS_PERFETTO_FUNC
  // can attribute their work without being passed any context.
  //
  // It is a per-thread stack: a module blocked in a non-isolated tbb::parallel_for
  // can have another module's task stolen onto its thread, nested inside it.
  struct ModuleContext {
    static constexpr unsigned kNoStream = ~0u;

    // Module label (owned by its ModuleDescription); nullptr if not traced.
    const char* label = nullptr;
    unsigned streamId = kNoStream;
  };

  // Returns false if the stack is full; the entry is then not recorded, but
  // push/pop stay balanced.
  bool pushModuleContext(ModuleContext const& ctx) noexcept;
  // Returns the popped entry (empty if it was not recorded).
  ModuleContext popModuleContext() noexcept;
  ModuleContext const& currentModuleContext() noexcept;

  class ModuleContextGuard {
  public:
    explicit ModuleContextGuard(ModuleContext const& ctx) noexcept { pushModuleContext(ctx); }
    ~ModuleContextGuard() noexcept { popModuleContext(); }
    ModuleContextGuard(ModuleContextGuard const&) = delete;
    ModuleContextGuard& operator=(ModuleContextGuard const&) = delete;
  };

  // The context is not inherited by threads a module spawns work on. Wrap such a
  // body to run it under the caller's context, e.g.
  //   tbb::parallel_for(range, cms::perfetto::withModuleContext([&](auto const& r) { ... }));
  template <class F>
  auto withModuleContext(F&& f) {
    return [ctx = currentModuleContext(), fn = std::decay_t<F>(std::forward<F>(f))](auto&&... args) mutable {
      ModuleContextGuard guard(ctx);
      return fn(std::forward<decltype(args)>(args)...);
    };
  }

}  // namespace cms::perfetto

#endif  // PerfTools_Perfetto_interface_CMSSWPerfettoModuleContext_h

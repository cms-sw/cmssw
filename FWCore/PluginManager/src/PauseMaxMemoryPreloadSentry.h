#ifndef FWCore_PluginManager_src_PauseMaxMemoryPreloadSentry_h
#define FWCore_PluginManager_src_PauseMaxMemoryPreloadSentry_h

namespace edm {
  class PauseMaxMemoryPreloadSentry {
  public:
    PauseMaxMemoryPreloadSentry();
    ~PauseMaxMemoryPreloadSentry();

    PauseMaxMemoryPreloadSentry(const PauseMaxMemoryPreloadSentry&) = delete;
    PauseMaxMemoryPreloadSentry(PauseMaxMemoryPreloadSentry&&) = delete;
    PauseMaxMemoryPreloadSentry& operator=(const PauseMaxMemoryPreloadSentry&) = delete;
    PauseMaxMemoryPreloadSentry& operator=(PauseMaxMemoryPreloadSentry&&) = delete;
  };
}  // namespace edm

#endif

#ifndef FWCore_Framework_CurrentModuleOnThread_h
#define FWCore_Framework_CurrentModuleOnThread_h

/** \class edm::CurrentModuleOnThread

\author W. David Dagenhart, created 30 August, 2013

*/

namespace edm {

  class ModuleCallingContext;
  class ModuleContextSentry;

  class CurrentModuleOnThread {
  public:
    static ModuleCallingContext const* getCurrentModuleOnThread() {
      return currentModuleOnThread_;
    }
  private:
    friend class ModuleContextSentry;
    static void setCurrentModuleOnThread(ModuleCallingContext const* v) {
      currentModuleOnThread_ = v;
    }

    static thread_local ModuleCallingContext const* currentModuleOnThread_;
  };
}
#endif

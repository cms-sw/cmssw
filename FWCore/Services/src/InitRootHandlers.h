#ifndef FWCore_Services_InitRootHandlers_h
#define FWCore_Services_InitRootHandlers_h

#include <memory>
#include "FWCore/Utilities/interface/RootHandlers.h"

namespace std {
  class thread;
}

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
  class ActivityRegistry;

  namespace service {
    class InitRootHandlers : public RootHandlers {

      friend int cmssw_stacktrace(void *);

    public:
      explicit InitRootHandlers(ParameterSet const& pset, ActivityRegistry& iReg);
      virtual ~InitRootHandlers();

      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void stacktraceFromThread();

    private:
      static char *const *getPstackArgv();
      virtual void enableWarnings_() override;
      virtual void ignoreWarnings_() override;
      virtual void willBeUsingThreads() override;
      virtual void initializeThisThreadForUse() override;

      void cachePidInfoHandler(unsigned int, unsigned int) {cachePidInfo();}
      void cachePidInfo();
      static void stacktraceHelperThread();

      static const int pidStringLength_ = 200;
      static char pidString_[pidStringLength_];
      static char * const pstackArgv_[];
      static int parentToChild_[2];
      static int childToParent_[2];
      static std::unique_ptr<std::thread> helperThread_;
      bool unloadSigHandler_;
      bool resetErrHandler_;
      bool loadAllDictionaries_;
      bool autoLibraryLoader_;
      std::shared_ptr<const void> sigBusHandler_;
      std::shared_ptr<const void> sigSegvHandler_;
      std::shared_ptr<const void> sigIllHandler_;
    };

    inline
    bool isProcessWideService(InitRootHandlers const*) {
      return true;
    }

  }  // end of namespace service
}  // end of namespace edm

#endif // InitRootHandlers_H

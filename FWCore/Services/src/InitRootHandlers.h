#ifndef FWCore_Services_InitRootHandlers_h
#define FWCore_Services_InitRootHandlers_h

#include "FWCore/Utilities/interface/RootHandlers.h"

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;

  namespace service {
    class InitRootHandlers : public RootHandlers {

    public:
      explicit InitRootHandlers(ParameterSet const& pset);
      virtual ~InitRootHandlers();

      inline
      bool isProcessWideService(InitRootHandlers const*) {
        return true;
      }

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      virtual void disableErrorHandler_();
      virtual void enableErrorHandler_();
      bool unloadSigHandler_;
      bool resetErrHandler_;
      bool autoLibraryLoader_;
    };
  }  // end of namespace service
}  // end of namespace edm

#endif // InitRootHandlers_H

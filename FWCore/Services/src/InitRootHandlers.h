#ifndef FWCore_Services_InitRootHandlers_h
#define FWCore_Services_InitRootHandlers_h

#include <memory>
#include "FWCore/Utilities/interface/RootHandlers.h"

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;

  namespace service {
    class InitRootHandlers : public RootHandlers {

    public:
      explicit InitRootHandlers(ParameterSet const& pset);
      virtual ~InitRootHandlers();

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      virtual void enableWarnings_() override;
      virtual void ignoreWarnings_() override;
      bool unloadSigHandler_;
      bool resetErrHandler_;
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

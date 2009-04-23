#ifndef FWCore_Services_InitRootHandlers_h
#define FWCore_Services_InitRootHandlers_h

#include "FWCore/Utilities/interface/RootHandlers.h"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
}

namespace edm {
namespace service {
class InitRootHandlers : public RootHandlers
{

public:
  InitRootHandlers (edm::ParameterSet const& pset, edm::ActivityRegistry & activity);
  virtual ~InitRootHandlers ();
  void postEndJob();

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

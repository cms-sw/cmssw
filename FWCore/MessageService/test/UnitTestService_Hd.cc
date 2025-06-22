#ifndef EDM_ML_DEBUG
#define EDM_ML_DEBUG
#endif

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edmtest {
  class UnitTestService_Hd {
  public:
    UnitTestService_Hd(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
      iRegistry.watchPreSourceNextTransition(
          []() { LogDebug("cat_S") << "Message from watchPreSourceNextTransition"; });
    }
  };
}  // namespace edmtest

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(edmtest::UnitTestService_Hd);

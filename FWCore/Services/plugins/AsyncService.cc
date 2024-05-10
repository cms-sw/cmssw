#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <atomic>

namespace edm::service {
  class AsyncService : public Async {
  public:
    AsyncService(ParameterSet const& iConfig, ActivityRegistry& iRegistry);

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void ensureAllowed() const final;

    std::atomic<bool> allowed_ = true;
  };

  AsyncService::AsyncService(ParameterSet const& iConfig, ActivityRegistry& iRegistry) {
    iRegistry.watchPreSourceEarlyTermination([this](TerminationOrigin) { allowed_ = false; });
    iRegistry.watchPreGlobalEarlyTermination([this](GlobalContext const&, TerminationOrigin) { allowed_ = false; });
    iRegistry.watchPreStreamEarlyTermination([this](StreamContext const&, TerminationOrigin) { allowed_ = false; });
    iRegistry.watchPostEndJob([this]() { allowed_ = false; });
  }

  void AsyncService::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    descriptions.addDefault(desc);
  }

  void AsyncService::ensureAllowed() const {
    if (not allowed_) {
      cms::Exception ex("AsyncCallNotAllowed");
      ex.addContext("Calling Async::run()");
      ex << "Framework is shutting down, further run() calls are not allowed";
      throw ex;
    }
  }
}  // namespace edm::service

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::AsyncService;
using AsyncMaker = edm::serviceregistry::AllArgsMaker<edm::Async, AsyncService>;
DEFINE_FWK_SERVICE_MAKER(AsyncService, AsyncMaker);

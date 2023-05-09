#include "WaitingService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

namespace edmtest {
  namespace test_acquire {

    WaitingService::WaitingService(edm::ParameterSet const& pset, edm::ActivityRegistry& iRegistry)
        : count_(0),
          numberOfStreamsToAccumulate_(pset.getUntrackedParameter<unsigned int>("streamsToAccumulate", 8)),
          secondsToWaitForWork_(pset.getUntrackedParameter<unsigned int>("secondsToWaitForWork", 1)) {
      iRegistry.watchPreallocate(this, &WaitingService::preallocate);
      iRegistry.watchPostEndJob(this, &WaitingService::postEndJob);
    }

    WaitingService::~WaitingService() {
      if (server_) {
        server_->stop();
      }
    }

    void WaitingService::preallocate(edm::service::SystemBounds const&) {
      caches_.resize(count_.load());
      server_ = std::make_unique<test_acquire::WaitingServer>(
          count_.load(), numberOfStreamsToAccumulate_, secondsToWaitForWork_);
      server_->start();
    }

    void WaitingService::postEndJob() {
      if (server_) {
        server_->stop();
      }
      server_.reset();
    }
  }  // namespace test_acquire
}  // namespace edmtest

using edmtest::test_acquire::WaitingService;
DEFINE_FWK_SERVICE(WaitingService);

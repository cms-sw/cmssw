#ifndef FWCore_Integration_WaitingService_h
#define FWCore_Integration_WaitingService_h

#include "WaitingServer.h"

#include <atomic>
#include <memory>
#include <vector>

namespace edm {
  class ActivityRegistry;
  class ParameterSet;
  namespace service {
    class SystemBounds;
  }
}  // namespace edm

namespace edmtest {
  namespace test_acquire {

    class Token {
    public:
      Token(unsigned int v) : value_(v) {}
      unsigned int value() const { return value_; }

    private:
      unsigned int value_;
    };

    class WaitingService {
    public:
      WaitingService(edm::ParameterSet const& pset, edm::ActivityRegistry& iRegistry);
      ~WaitingService();

      Token getToken() { return Token(count_.fetch_add(1)); }

      void preallocate(edm::service::SystemBounds const&);

      Cache* getCache(Token const& token) { return &caches_[token.value()]; }

      void requestValuesAsync(Token const& token,
                              std::vector<int> const* iIn,
                              std::vector<int>* iOut,
                              edm::WaitingTaskWithArenaHolder& holder) {
        server_->requestValuesAsync(token.value(), iIn, iOut, holder);
      }

      void postEndJob();

    private:
      std::atomic<unsigned int> count_;
      std::vector<Cache> caches_;
      std::unique_ptr<WaitingServer> server_;
      const unsigned int numberOfStreamsToAccumulate_;
      const unsigned int secondsToWaitForWork_;
    };
  }  // namespace test_acquire
}  // namespace edmtest
#endif

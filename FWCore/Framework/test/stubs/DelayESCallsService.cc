#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <cassert>

namespace edmtest {
  class DelayESCallsService {
  public:
    DelayESCallsService(edm::ParameterSet const& pset, edm::ActivityRegistry& reg)
        : delay_(pset.getUntrackedParameter<unsigned int>("delay")), count_(0) {
      syncValue_ = edm::IOVSyncValue(edm::EventID(
          pset.getUntrackedParameter<unsigned int>("run"), pset.getUntrackedParameter<unsigned int>("lumi"), 0));
      reg.preESSyncIOVSignal_.connect([this](edm::IOVSyncValue const& iSync) {
        auto c = ++count_;
        assert(c == 1);
        if (iSync == syncValue_)
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_));
      });
      reg.postESSyncIOVSignal_.connect([this](edm::IOVSyncValue const&) {
        --count_;
        assert(count_ == 0);
      });
    }

  private:
    edm::IOVSyncValue syncValue_;
    unsigned int delay_;
    std::atomic<unsigned int> count_;
  };
}  // namespace edmtest

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(edmtest::DelayESCallsService);

#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <condition_variable>
#include <mutex>

namespace edmtest {
  class AsyncServiceTesterService {
  public:
    AsyncServiceTesterService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) : continue_{false} {
      if (iConfig.getParameter<bool>("watchEarlyTermination")) {
        iRegistry.watchPreSourceEarlyTermination([this](edm::TerminationOrigin) { release(); });
        iRegistry.watchPreGlobalEarlyTermination(
            [this](edm::GlobalContext const&, edm::TerminationOrigin) { release(); });
        iRegistry.watchPreStreamEarlyTermination(
            [this](edm::StreamContext const&, edm::TerminationOrigin) { release(); });
      }
      if (iConfig.getParameter<bool>("watchStreamEndRun")) {
        // StreamEndRun is the last stream transition in the data
        // processing that does not depend on any global end
        // transition
        iRegistry.watchPostStreamEndRun([this](edm::StreamContext const&) { release(); });
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("watchEarlyTermination", false)
          ->setComment("If true, watch EarlyTermination signals to signal the waiters");
      desc.add("watchStreamEndRun", false)->setComment("If true, watch StreamEndRun signals to signal the waiters");
      descriptions.addDefault(desc);
    }

    void wait() {
      std::unique_lock lk(mutex_);
      if (continue_)
        return;
      cond_.wait(lk, [this]() { return continue_; });
    }

    bool stillWaiting() const {
      std::unique_lock lk(mutex_);
      return not continue_;
    }

  private:
    void release() {
      std::unique_lock lk(mutex_);
      continue_ = true;
      cond_.notify_all();
    }

    mutable std::mutex mutex_;
    std::condition_variable cond_;
    CMS_THREAD_GUARD(mutex_) bool continue_;
  };

  struct AsyncServiceTesterCache {
    struct RunGuard {
      RunGuard(std::atomic<int>* c) : calls(c) {}
      ~RunGuard() {
        if (calls) {
          --(*calls);
        }
      }
      void release() { calls = nullptr; }
      RunGuard(RunGuard const&) = delete;
      RunGuard& operator=(RunGuard const&) = delete;
      RunGuard(RunGuard&& o) = delete;
      RunGuard& operator=(RunGuard&&) = delete;

      std::atomic<int>* calls = nullptr;
    };

    RunGuard makeRunCallGuard(int inc) const {
      outstandingRunCalls += inc;
      return RunGuard(&outstandingRunCalls);
    }

    mutable std::atomic<int> outstandingRunCalls = 0;
  };

  class AsyncServiceTester
      : public edm::stream::EDProducer<edm::ExternalWork, edm::GlobalCache<AsyncServiceTesterCache>> {
  public:
    AsyncServiceTester(edm::ParameterSet const& iConfig, AsyncServiceTesterCache const*) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

    static auto initializeGlobalCache(edm::ParameterSet const&) { return std::make_unique<AsyncServiceTesterCache>(); }

    void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      if (status_ != 0) {
        throw cms::Exception("Assert") << "In acquire: status_ was " << status_ << ", expected 0";
      }
      edm::Service<edm::Async> as;
      auto callGuard = globalCache()->makeRunCallGuard(1);
      as->runAsync(
          std::move(holder),
          [this]() {
            auto callGuard = globalCache()->makeRunCallGuard(0);
            if (status_ != 0) {
              throw cms::Exception("Assert") << "In async function: status_ was " << status_ << ", expected 0";
            }
            ++status_;
          },
          []() { return "Calling AsyncServiceTester::acquire()"; });
      callGuard.release();
    }

    void produce(edm::Event&, edm::EventSetup const&) final {
      if (status_ != 1) {
        throw cms::Exception("Assert") << "In analyze: status_ was " << status_ << ", expected 1";
      }
      status_ = 0;
    }

    static void globalEndJob(AsyncServiceTesterCache* cache) {
      if (cache->outstandingRunCalls != 0) {
        throw cms::Exception("Assert") << "In globalEndJob: " << cache->outstandingRunCalls
                                       << " runAsync() calls outstanding, expected 0";
      }
    }

  private:
    std::atomic<int> status_ = 0;
  };

  class AsyncServiceWaitingTester : public edm::stream::EDProducer<edm::ExternalWork,
                                                                   edm::GlobalCache<AsyncServiceTesterCache>,
                                                                   edm::stream::WatchLuminosityBlocks,
                                                                   edm::stream::WatchRuns> {
  public:
    AsyncServiceWaitingTester(edm::ParameterSet const& iConfig, AsyncServiceTesterCache const*)
        : throwingStream_(iConfig.getUntrackedParameter<unsigned int>("throwingStream")),
          waitEarlyTermination_(iConfig.getUntrackedParameter<bool>("waitEarlyTermination")),
          waitStreamEndRun_(iConfig.getUntrackedParameter<bool>("waitStreamEndRun")) {
      if (not waitEarlyTermination_ and not waitStreamEndRun_) {
        throw cms::Exception("Configuration")
            << "One of 'waitEarlyTermination' and 'waitStreamEndRun' must be set to True, both were False";
      }
      if (waitEarlyTermination_ and waitStreamEndRun_) {
        throw cms::Exception("Configuration")
            << "Only one of 'waitEarlyTermination' and 'waitStreamEndRun' can be set to True, both were True";
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<unsigned int>("throwingStream")
          ->setComment("ID of the stream where another module throws an exception");
      desc.addUntracked("waitEarlyTermination", false)
          ->setComment(
              "If true, use AsyncServiceTesterService in streams other than 'throwingStream' to wait launching the "
              "async activity until an early termination signal has been issued");
      desc.addUntracked("waitStreamEndRun", false)
          ->setComment(
              "If true, wait in the async activity in streams other than 'throwingStream' until one stream has reached "
              "streamEndRun");
      descriptions.addDefault(desc);
      descriptions.setComment("One of 'waitEarlyTermination' and 'waitStreamEndRun' must be set to 'True'");
    }

    static auto initializeGlobalCache(edm::ParameterSet const&) { return std::make_unique<AsyncServiceTesterCache>(); }

    void beginStream(edm::StreamID id) { streamId_ = id; }

    void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      bool const waitOnThisStream = *streamId_ != throwingStream_;
      AsyncServiceTesterService* testService = nullptr;
      if (waitOnThisStream) {
        edm::Service<AsyncServiceTesterService> tsh;
        testService = &*tsh;
        if (waitEarlyTermination_)
          testService->wait();
      }
      if (status_ != 0) {
        throw cms::Exception("Assert") << "In acquire: status_ was " << status_ << ", expected 0";
      }
      edm::Service<edm::Async> as;
      auto callGuard = globalCache()->makeRunCallGuard(1);
      as->runAsync(
          std::move(holder),
          [this, testService]() {
            auto callGuard = globalCache()->makeRunCallGuard(0);
            if (testService and waitStreamEndRun_) {
              testService->wait();
            }

            if (status_ != 0) {
              throw cms::Exception("Assert") << "In async function: status_ was " << status_ << ", expected 0";
            }
            ++status_;
          },
          []() { return "Calling AsyncServiceTester::acquire()"; });
      callGuard.release();
    }

    void produce(edm::Event&, edm::EventSetup const&) final {
      if (status_ != 1) {
        throw cms::Exception("Assert") << "In analyze: status_ was " << status_ << ", expected 1";
      }
      status_ = 0;
    }

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final {
      if (edm::Service<AsyncServiceTesterService>()->stillWaiting() and *streamId_ != throwingStream_) {
        throw cms::Exception("Assert") << "In endLuminosityBlock for stream " << *streamId_
                                       << " that is different from the throwing stream " << throwingStream_
                                       << " while the waits have not been signaled";
      }
    }

    void endRun(edm::Run const&, edm::EventSetup const&) final {
      if (edm::Service<AsyncServiceTesterService>()->stillWaiting() and *streamId_ != throwingStream_) {
        throw cms::Exception("Assert") << "In endRun for stream " << *streamId_
                                       << " that is different from the throwing stream " << throwingStream_
                                       << " while the waits have not been signaled";
      }
    }

    static void globalEndJob(AsyncServiceTesterCache* cache) {
      if (cache->outstandingRunCalls != 0) {
        throw cms::Exception("Assert") << "In globalEndJob: " << cache->outstandingRunCalls
                                       << " runAsync() calls outstanding, expected 0";
      }
    }

  private:
    std::atomic<int> status_ = 0;
    std::optional<edm::StreamID> streamId_;
    unsigned int const throwingStream_;
    bool const waitEarlyTermination_;
    bool const waitStreamEndRun_;
  };
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::AsyncServiceTester);
DEFINE_FWK_MODULE(edmtest::AsyncServiceWaitingTester);

DEFINE_FWK_SERVICE(edmtest::AsyncServiceTesterService);

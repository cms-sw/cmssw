#if !defined(TEST_WORKER)
#define TEST_WORKER
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>
#include <stdio.h>
#include <cassert>
#include <thread>

enum class WorkerType { kStandard, kOKTimeout, kStartupTimeout };

int worker(int argc, char** argv, WorkerType iType) {
  using namespace edm::shared_memory;

  using namespace std::chrono_literals;
  if (iType == WorkerType::kStartupTimeout) {
    //Take too long before openning the worker channel
    std::this_thread::sleep_for(20s);
  }

  assert(argc == 3);
  WorkerChannel channel(argv[1], argv[2]);

  if (iType == WorkerType::kOKTimeout) {
    //simulate long time setting up
    std::this_thread::sleep_for(20s);
  }
  //std::cerr<<"worker setup\n";
  channel.workerSetupDone();

  if (iType == WorkerType::kOKTimeout) {
    std::this_thread::sleep_for(20s);
  }
  int transitionCount = 0;
  channel.handleTransitions([&](edm::Transition iTransition, unsigned long long iTransitionID) {
    if (iType == WorkerType::kOKTimeout) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(20s);
    }
    switch (transitionCount) {
      case 0: {
        if (iTransition != edm::Transition::Event) {
          throw cms::Exception("BadValue") << "wrong transition received " << static_cast<int>(iTransition);
        }
        if (iTransitionID != 2ULL) {
          throw cms::Exception("BadValue") << "wrong transitionID received " << static_cast<int>(iTransitionID);
        }

        if (channel.toWorkerBufferInfo()->index_ != 0) {
          throw cms::Exception("BadValue")
              << "wrong toWorkerBufferInfo index received " << static_cast<int>(channel.toWorkerBufferInfo()->index_);
        }
        if (channel.toWorkerBufferInfo()->identifier_ != 0) {
          throw cms::Exception("BadValue")
              << "wrong toWorkerBufferInfo identifier received " << channel.toWorkerBufferInfo()->identifier_;
        }
        *channel.fromWorkerBufferInfo() = {1, 1};
        channel.shouldKeepEvent(true);
        break;
      }

      case 1: {
        if (iTransition != edm::Transition::Event) {
          throw cms::Exception("BadValue") << "wrong transition received " << static_cast<int>(iTransition);
        }
        if (iTransitionID != 3ULL) {
          throw cms::Exception("BadValue") << "wrong transitionID received " << static_cast<int>(iTransitionID);
        }

        if (channel.toWorkerBufferInfo()->index_ != 1) {
          throw cms::Exception("BadValue")
              << "wrong toWorkerBufferInfo index received " << static_cast<int>(channel.toWorkerBufferInfo()->index_);
        }
        if (channel.toWorkerBufferInfo()->identifier_ != 1) {
          throw cms::Exception("BadValue")
              << "wrong toWorkerBufferInfo identifier received " << channel.toWorkerBufferInfo()->identifier_;
        }
        *channel.fromWorkerBufferInfo() = {2, 0};
        channel.shouldKeepEvent(false);
        break;
      }

      case 2: {
        if (iTransition != edm::Transition::EndLuminosityBlock) {
          throw cms::Exception("BadValue") << "wrong transition received " << static_cast<int>(iTransition);
        }
        if (iTransitionID != 1ULL) {
          throw cms::Exception("BadValue") << "wrong transitionID received " << static_cast<int>(iTransitionID);
        }
        break;
      }
      default: {
        throw cms::Exception("MissingStop") << "stopRequested not set";
      }
    }
    ++transitionCount;
  });
  if (transitionCount != 3) {
    throw cms::Exception("MissingStop") << "stop requested too soon " << transitionCount;
  }
  return 0;
}
#endif

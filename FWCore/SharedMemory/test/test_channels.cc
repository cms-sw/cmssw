#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>
#include <stdio.h>
#include <cassert>
namespace {
  int controller(int argc, char** argv) {
    using namespace edm::shared_memory;

    ControllerChannel channel("TestChannel", 0);

    //Pipe has to close AFTER we tell the worker to stop
    auto closePipe = [](FILE* iFile) { pclose(iFile); };
    std::unique_ptr<FILE, decltype(closePipe)> pipe(nullptr, closePipe);

    auto stopWorkerCmd = [](ControllerChannel* iChannel) { iChannel->stopWorker(); };
    std::unique_ptr<ControllerChannel, decltype(stopWorkerCmd)> stopWorkerGuard(&channel, stopWorkerCmd);

    {
      std::string command(argv[0]);
      command += " ";
      command += channel.sharedMemoryName();
      command += " ";
      command += channel.uniqueID();
      //make sure output is flushed before popen does any writing
      fflush(stdout);
      fflush(stderr);

      channel.setupWorker([&]() {
        pipe.reset(popen(command.c_str(), "w"));

        if (not pipe) {
          throw cms::Exception("PipeFailed") << "pipe failed to open " << command;
        }
      });
    }
    {
      *channel.toWorkerBufferIndex() = 0;
      auto result = channel.doTransition(
          [&]() {
            if (*channel.fromWorkerBufferIndex() != 1) {
              throw cms::Exception("BadValue")
                  << "wrong value of fromWorkerBufferIndex " << *channel.fromWorkerBufferIndex();
            }
            if (not channel.shouldKeepEvent()) {
              throw cms::Exception("BadValue") << "told not to keep event";
            }
          },
          edm::Transition::Event,
          2);
      if (not result) {
        throw cms::Exception("TimeOut") << "doTransition timed out";
      }
    }
    {
      *channel.toWorkerBufferIndex() = 1;
      auto result = channel.doTransition(
          [&]() {
            if (*channel.fromWorkerBufferIndex() != 0) {
              throw cms::Exception("BadValue")
                  << "wrong value of fromWorkerBufferIndex " << *channel.fromWorkerBufferIndex();
            }
            if (channel.shouldKeepEvent()) {
              throw cms::Exception("BadValue") << "told to keep event";
            }
          },
          edm::Transition::Event,
          3);
      if (not result) {
        throw cms::Exception("TimeOut") << "doTransition timed out";
      }
    }

    {
      auto result = channel.doTransition([&]() {}, edm::Transition::EndLuminosityBlock, 1);
      if (not result) {
        throw cms::Exception("TimeOut") << "doTransition timed out";
      }
    }

    //std::cout <<"controller going to stop"<<std::endl;
    return 0;
  }

  int worker(int argc, char** argv) {
    using namespace edm::shared_memory;

    assert(argc == 3);
    WorkerChannel channel(argv[1], argv[2]);

    //std::cerr<<"worker setup\n";
    channel.workerSetupDone();

    int transitionCount = 0;
    channel.handleTransitions([&](edm::Transition iTransition, unsigned long long iTransitionID) {
      switch (transitionCount) {
        case 0: {
          if (iTransition != edm::Transition::Event) {
            throw cms::Exception("BadValue") << "wrong transition received " << static_cast<int>(iTransition);
          }
          if (iTransitionID != 2ULL) {
            throw cms::Exception("BadValue") << "wrong transitionID received " << static_cast<int>(iTransitionID);
          }

          if (*channel.toWorkerBufferIndex() != 0) {
            throw cms::Exception("BadValue") << "wrong toWorkerBufferIndex received " << *channel.toWorkerBufferIndex();
          }
          *channel.fromWorkerBufferIndex() = 1;
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

          if (*channel.toWorkerBufferIndex() != 1) {
            throw cms::Exception("BadValue") << "wrong toWorkerBufferIndex received " << *channel.toWorkerBufferIndex();
          }
          *channel.fromWorkerBufferIndex() = 0;
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
  const char* jobType(bool isWorker) {
    if (isWorker) {
      return "Worker";
    }
    return "Controller";
  }

}  // namespace

int main(int argc, char** argv) {
  bool isWorker = true;
  int retValue = 0;
  try {
    if (argc > 1) {
      retValue = worker(argc, argv);
    } else {
      isWorker = false;
      retValue = controller(argc, argv);
    }
  } catch (std::exception const& iException) {
    std::cerr << "Caught exception\n" << iException.what() << "\n";
    if (isWorker) {
      std::cerr << "in worker\n";
    } else {
      std::cerr << "in controller\n";
    }
    return 1;
  }
  if (0 == retValue) {
    std::cout << jobType(isWorker) << " success" << std::endl;
  } else {
    std::cout << jobType(isWorker) << " failed" << std::endl;
  }
  return 0;
}

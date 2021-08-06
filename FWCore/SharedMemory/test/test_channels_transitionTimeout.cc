#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>
#include <stdio.h>
#include <cassert>
#include <thread>

namespace {
  int controller(int argc, char** argv) {
    using namespace edm::shared_memory;

    ControllerChannel channel("TestChannel", 0, 5);

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
      *channel.toWorkerBufferInfo() = {0, 0};
      auto result = channel.doTransition([&]() { /*job will fail before calling*/ }, edm::Transition::Event, 2);
      if (not result) {
        //this should happen as we should time out
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

    channel.handleTransitions(
        [&](edm::Transition iTransition, unsigned long long iTransitionID) { throw cms::Exception("BAD"); });
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
  } catch (cms::Exception const& iException) {
    if (iException.category() != "TimeOut") {
      std::cerr << "Caught exception\n" << iException.what() << "\n";
      return 1;
    } else {
      std::cout << "expected failure occurred\n";
      return 0;
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
  return 1;
}

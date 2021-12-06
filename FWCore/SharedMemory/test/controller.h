#if !defined(TEST_CONTROLLER)
#define TEST_CONTROLLER
#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>
#include <stdio.h>
int controller(int argc, char** argv, unsigned int iTimeout) {
  using namespace edm::shared_memory;

  ControllerChannel channel("TestChannel", 0, iTimeout);

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
    auto result = channel.doTransition(
        [&]() {
          if (channel.fromWorkerBufferInfo()->index_ != 1) {
            throw cms::Exception("BadValue") << "wrong index value of fromWorkerBufferInfo "
                                             << static_cast<int>(channel.fromWorkerBufferInfo()->index_);
          }
          if (channel.fromWorkerBufferInfo()->identifier_ != 1) {
            throw cms::Exception("BadValue")
                << "wrong identifier value of fromWorkerBufferInfo " << channel.fromWorkerBufferInfo()->identifier_;
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
    *channel.toWorkerBufferInfo() = {1, 1};
    auto result = channel.doTransition(
        [&]() {
          if (channel.fromWorkerBufferInfo()->index_ != 0) {
            throw cms::Exception("BadValue") << "wrong index value of fromWorkerBufferInfo "
                                             << static_cast<int>(channel.fromWorkerBufferInfo()->index_);
          }
          if (channel.fromWorkerBufferInfo()->identifier_ != 2) {
            throw cms::Exception("BadValue")
                << "wrong identifier value of fromWorkerBufferInfo " << channel.fromWorkerBufferInfo()->identifier_;
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

#endif

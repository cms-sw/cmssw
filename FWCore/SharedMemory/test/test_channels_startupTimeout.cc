#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>
#include <stdio.h>
#include <cassert>
#include <thread>

#include "controller.h"
#include "worker.h"
namespace {
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
      retValue = worker(argc, argv, WorkerType::kStartupTimeout);
    } else {
      isWorker = false;
      retValue = controller(argc, argv, 5);
    }
  } catch (cms::Exception const& iException) {
    if (iException.category() != "ExternalFailure") {
      throw;
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

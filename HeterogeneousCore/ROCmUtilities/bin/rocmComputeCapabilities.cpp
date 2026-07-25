// C/C++ standard headers
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// ROCm headers
#include <hip/hip_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"
#ifndef HIP_CHECK
#define HIP_CHECK hipCheck
#endif

// local headers
#include "isRocmDeviceSupported.h"

namespace {

  // print a short usage message
  void printUsage(std::string_view name) {
    std::cout << "Usage: " << name << " [-u|--uuid] [-h|--help]\n\n"
              << "Print the index, compute capability and name of each visible ROCm device.\n\n"
              << "Options:\n"
              << "  -u, --uuid      print the device UUIDs instead of indices\n"
              << "  -h, --help      print this help message and exit\n";
  }

  // query the device UUID
  std::string uuid(int device) {
    hipUUID uuid;
    HIP_CHECK(hipDeviceGetUuid(&uuid, device));
    return std::string("GPU-") + std::string(uuid.bytes, std::size(uuid.bytes));
  }

}  // namespace

int main(int argc, char** argv) {
  bool useuuid = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--uuid" or arg == "-u") {
      useuuid = true;
    } else if (arg == "--help" or arg == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "rocmComputeCapabilities: unrecognised option '" << arg << "'\n";
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  int devices = 0;
  hipError_t status = hipGetDeviceCount(&devices);
  if (status != hipSuccess) {
    std::cerr << "rocmComputeCapabilities: " << hipGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < devices; ++i) {
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, i));
    if (useuuid) {
      std::cout << uuid(i);
    } else {
      std::cout << std::setw(4) << i;
    }
    std::cout << "    " << std::setw(8) << properties.gcnArchName << "    " << properties.name;
    if (not isRocmDeviceSupported(i)) {
      std::cout << " (unsupported)";
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}

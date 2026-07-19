// C/C++ standard headers
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#ifndef CUDA_CHECK
#define CUDA_CHECK cudaCheck
#endif
#include "HeterogeneousCore/CUDAUtilities/interface/nvmlCheck.h"
#ifndef NVML_CHECK
#define NVML_CHECK nvmlCheck
#endif

// local headers
#include "isCudaDeviceSupported.h"

namespace {

  // print a short usage message
  void printUsage(std::string_view name) {
    std::cout << "Usage: " << name << " [-u|--uuid] [-v|--verbose] [-h|--help]\n\n"
              << "Print the index, compute capability and name of each visible CUDA device.\n\n"
              << "Options:\n"
              << "  -u, --uuid      print the device UUIDs instead of indices\n"
              << "  -v, --verbose   print detailed properties for each device\n"
              << "  -h, --help      print this help message and exit\n";
  }

  std::string_view yesno(int value) { return value ? "yes" : "no"; }

  // format the PCI location as domain:bus:device, like nvidia-smi
  std::string pciId(cudaDeviceProp const& properties) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(4) << properties.pciDomainID << ':' << std::setw(2)
        << properties.pciBusID << ':' << std::setw(2) << properties.pciDeviceID << ".0";
    return out.str();
  }

  // query the device UUID
  std::string uuid(int device) {
    CUdevice cu_device{};
    CUDA_CHECK(cuDeviceGet(&cu_device, device));
    CUuuid cu_uuid{};
    CUDA_CHECK(cuDeviceGetUuid(&cu_uuid, cu_device));
    nvmlUUID_t nv_uuid{.version = nvmlUUID_v1, .type = NVML_UUID_TYPE_BINARY, .value{.bytes{}}};
    std::memcpy(nv_uuid.value.bytes, cu_uuid.bytes, NVML_DEVICE_UUID_BINARY_LEN);
    nvmlDevice_t nv_device;
    NVML_CHECK(nvmlDeviceGetHandleByUUIDV(&nv_uuid, &nv_device));
    char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE];
    NVML_CHECK(nvmlDeviceGetUUID(nv_device, uuid, sizeof(uuid)));
    return std::string(uuid);
  }

  // print the detailed properties of a device, indented under its summary line
  void printDetails(int device, cudaDeviceProp const& properties) {
    // query CUDA attributes by ordinal
    auto attribute = [device](cudaDeviceAttr attr) {
      int value = 0;
      cudaDeviceGetAttribute(&value, attr, device);
      return value;
    };

    // compose a value and an optional unit into a single string
    auto value = [](auto const& v, std::string_view unit = "") {
      std::ostringstream out;
      out << v << unit;
      return out.str();
    };

    // compose the three components of a dimension into "x X y X z"
    auto dimensions = [](const int* d) {
      std::ostringstream out;
      out << d[0] << " x " << d[1] << " x " << d[2];
      return out.str();
    };

    std::vector<std::pair<std::string_view, std::string>> rows = {
        {"PCI device id:", pciId(properties)},
        {"UUID:", uuid(device)},
        {"integrated:", value(yesno(properties.integrated))},
        {"total global memory:", value(properties.totalGlobalMem >> 20, " MiB")},
        {"L2 cache size:", value(properties.l2CacheSize >> 10, " KiB")},
        {"shared memory per block:", value(properties.sharedMemPerBlock >> 10, " KiB")},
        {"shared memory per multiprocessor:", value(properties.sharedMemPerMultiprocessor >> 10, " KiB")},
        {"memory bus width:", value(properties.memoryBusWidth, " bits")},
        {"memory clock rate:", value(attribute(cudaDevAttrMemoryClockRate) / 1000, " MHz")},
        {"GPU clock rate:", value(attribute(cudaDevAttrClockRate) / 1000, " MHz")},
        {"multiprocessors:", value(properties.multiProcessorCount)},
        {"warp size:", value(properties.warpSize)},
        {"max threads per multiprocessor:", value(properties.maxThreadsPerMultiProcessor)},
        {"max resident threads:", value(properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor)},
        {"max threads per block:", value(properties.maxThreadsPerBlock)},
        {"max block dimensions:", dimensions(properties.maxThreadsDim)},
        {"max grid dimensions:", dimensions(properties.maxGridSize)},
        {"registers per block:", value(properties.regsPerBlock)},
        {"ECC enabled:", value(yesno(properties.ECCEnabled))},
        {"concurrent kernels:", value(yesno(properties.concurrentKernels))},
        {"async engine count:", value(properties.asyncEngineCount)},
        {"cooperative launch:", value(yesno(properties.cooperativeLaunch))},
        {"unified addressing:", value(yesno(properties.unifiedAddressing))},
        {"managed memory:", value(yesno(properties.managedMemory))},
        {"can map host memory:", value(yesno(properties.canMapHostMemory))},
    };

    // align the label column on the left and the value column on the right, so that
    // both the left and right margins of the indented block line up for every row
    std::size_t labelWidth = 0;
    std::size_t valueWidth = 0;
    for (auto const& [label, text] : rows) {
      labelWidth = std::max(labelWidth, label.size());
      valueWidth = std::max(valueWidth, text.size());
    }

    for (auto const& [label, text] : rows) {
      std::cout << "    " << std::left << std::setw(static_cast<int>(labelWidth)) << label << "  " << std::right
                << std::setw(static_cast<int>(valueWidth)) << text << "\n";
    }
    std::cout << '\n';
  }
}  // namespace

int main(int argc, char** argv) {
  bool useuuid = false;
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--verbose" or arg == "-v") {
      verbose = true;
    } else if (arg == "--uuid" or arg == "-u") {
      useuuid = true;
    } else if (arg == "--help" or arg == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "cudaComputeCapabilities: unrecognised option '" << arg << "'\n";
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  // initialise CUDA
  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    const char* error;
    CUDA_CHECK(cuGetErrorString(status, &error));
    std::cerr << "cudaComputeCapabilities: " << error << '\n';
    return EXIT_FAILURE;
  }

  // initialise NVML
  NVML_CHECK(nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS));

  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));

  for (int i = 0; i < devices; ++i) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);
    if (useuuid) {
      std::cout << uuid(i);
    } else {
      std::cout << std::setw(4) << i;
    }
    std::cout << "    " << std::setw(2) << properties.major << "." << properties.minor << "    " << properties.name;
    if (not isCudaDeviceSupported(i)) {
      std::cout << " (unsupported)";
    }
    std::cout << std::endl;

    if (verbose) {
      printDetails(i, properties);
    }
  }

  // shut down NVML
  NVML_CHECK(nvmlShutdown());

  return EXIT_SUCCESS;
}

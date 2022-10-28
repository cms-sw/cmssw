#ifndef HeterogeneousCore_CUDAUtilities_nvmlCheck_h
#define HeterogeneousCore_CUDAUtilities_nvmlCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

// CUDA headers
#include <nvml.h>

// CMSSW headers
#include "FWCore/Utilities/interface/Likely.h"

namespace cms {
  namespace cuda {

    [[noreturn]] inline void abortOnNvmlError(const char* file,
                                              int line,
                                              const char* cmd,
                                              const char* error,
                                              const char* message,
                                              std::string_view description = std::string_view()) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "nvmlCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (!description.empty())
        out << description << "\n";
      throw std::runtime_error(out.str());
    }

    inline bool nvmlCheck_(const char* file,
                           int line,
                           const char* cmd,
                           nvmlReturn_t result,
                           std::string_view description = std::string_view()) {
      if (LIKELY(result == NVML_SUCCESS))
        return true;

      std::string error = "NVML Error " + std::to_string(result);
      const char* message = nvmlErrorString(result);
      abortOnNvmlError(file, line, cmd, error.c_str(), message, description);
      return false;
    }
  }  // namespace cuda
}  // namespace cms

#define nvmlCheck(ARG, ...) (cms::cuda::nvmlCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_CUDAUtilities_nvmlCheck_h

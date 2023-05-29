#ifndef HeterogeneousCore_ROCmUtilities_hipCheck_h
#define HeterogeneousCore_ROCmUtilities_hipCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

// ROCm headers
#include <hip/hip_runtime.h>

// CMSSW headers
#include "FWCore/Utilities/interface/Likely.h"

namespace cms {
  namespace rocm {

    [[noreturn]] inline void abortOnError(const char* file,
                                          int line,
                                          const char* cmd,
                                          const char* error,
                                          const char* message,
                                          std::string_view description = std::string_view()) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "hipCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (!description.empty())
        out << description << "\n";
      throw std::runtime_error(out.str());
    }

    inline bool hipCheck_(const char* file,
                          int line,
                          const char* cmd,
                          hipError_t result,
                          std::string_view description = std::string_view()) {
      if (LIKELY(result == hipSuccess))
        return true;

      const char* error = hipGetErrorName(result);
      const char* message = hipGetErrorString(result);
      abortOnError(file, line, cmd, error, message, description);
      return false;
    }
  }  // namespace rocm
}  // namespace cms

#define hipCheck(ARG, ...) (cms::rocm::hipCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_ROCmUtilities_hipCheck_h

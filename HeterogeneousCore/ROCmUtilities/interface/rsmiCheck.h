#ifndef HeterogeneousCore_ROCmUtilities_rsmiCheck_h
#define HeterogeneousCore_ROCmUtilities_rsmiCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

// ROCm headers
#include <rocm_smi/rocm_smi.h>

// CMSSW headers
#include "FWCore/Utilities/interface/Likely.h"

namespace cms {
  namespace rocm {

    [[noreturn]] inline void abortOnRsmiError(const char* file,
                                              int line,
                                              const char* cmd,
                                              const char* error,
                                              const char* message,
                                              std::string_view description = std::string_view()) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "rsmiCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (!description.empty())
        out << description << "\n";
      throw std::runtime_error(out.str());
    }

    inline bool rsmiCheck_(const char* file,
                           int line,
                           const char* cmd,
                           rsmi_status_t result,
                           std::string_view description = std::string_view()) {
      if (LIKELY(result == RSMI_STATUS_SUCCESS))
        return true;

      std::string error = "ROCm SMI Error " + std::to_string(result);
      const char* message;
      rsmi_status_string(result, &message);
      abortOnRsmiError(file, line, cmd, error.c_str(), message, description);
      return false;
    }
  }  // namespace rocm
}  // namespace cms

#define rsmiCheck(ARG, ...) (cms::rocm::rsmiCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_ROCmUtilities_rsmiCheck_h

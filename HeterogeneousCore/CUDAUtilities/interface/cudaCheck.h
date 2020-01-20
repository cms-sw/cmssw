#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CMSSW headers
#include "FWCore/Utilities/interface/Likely.h"

namespace cms {
  namespace cuda {

    [[noreturn]] inline void abortOnCudaError(const char* file,
                                              int line,
                                              const char* cmd,
                                              const char* error,
                                              const char* message,
                                              const char* description = nullptr) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "cudaCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (description)
        out << description << "\n";
      throw std::runtime_error(out.str());
    }

    inline bool cudaCheck_(
        const char* file, int line, const char* cmd, CUresult result, const char* description = nullptr) {
      if (LIKELY(result == CUDA_SUCCESS))
        return true;

      const char* error;
      const char* message;
      cuGetErrorName(result, &error);
      cuGetErrorString(result, &message);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }

    inline bool cudaCheck_(
        const char* file, int line, const char* cmd, cudaError_t result, const char* description = nullptr) {
      if (LIKELY(result == cudaSuccess))
        return true;

      const char* error = cudaGetErrorName(result);
      const char* message = cudaGetErrorString(result);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }

  }  // namespace cuda
}  // namespace cms

#define cudaCheck(ARG, ...) (cms::cuda::cudaCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_CUDAUtilities_cudaCheck_h

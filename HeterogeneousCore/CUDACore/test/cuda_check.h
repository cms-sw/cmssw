#ifndef cuda_check_h
#define cuda_check_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda_runtime.h>

inline bool cuda_check__(const char* file, int line, const char* cmd, cudaError_t result) {
  if (__builtin_expect(result == cudaSuccess, true))
    return true;

  // reset the CUDA error flag
  (void) cudaGetLastError();

  // decode and print the error name and description
  const char* error = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);

  std::ostringstream out;
  out << "\n";
  out << file << ", line " << line << ":\n";
  out << "CUDA_CHECK(" << cmd << ");\n";
  out << error << ": " << message << "\n";
  std::cerr << out.str() << std::flush;

  return false;
}

#define CUDA_CHECK(ARG) (cuda_check__(__FILE__, __LINE__, #ARG, (ARG)))

#endif  // cuda_check_h



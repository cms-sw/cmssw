#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

  inline
  void printCudaErrorMessage(const char* file, int line, const char* cmd, const char* error, const char* message) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "cudaCheck(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    std::cerr << out.rdbuf() << std::endl;
  }

}

inline
bool cudaCheck_(const char* file, int line, const char* cmd, CUresult result)
{
    if (__builtin_expect(result == CUDA_SUCCESS, true))
        return true;

    const char* error;
    const char* message;
    cuGetErrorName(result, &error);
    cuGetErrorString(result, &message);
    printCudaErrorMessage(file, line, cmd, error, message);
    abort();
    return false;
}

inline
bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result)
{
    if (__builtin_expect(result == cudaSuccess, true))
        return true;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    printCudaErrorMessage(file, line, cmd, error, message);
    abort();
    return false;
}

#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

#endif // HeterogeneousCore_CUDAUtilities_cudaCheck_h

#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

inline
bool cudaCheck_(const char* file, int line, const char* cmd, CUresult result)
{
    //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
    if (result == CUDA_SUCCESS)
        return true;

    const char* error;
    const char* message;
    cuGetErrorName(result, &error);
    cuGetErrorString(result, &message);
    std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
    abort();
    return false;
}

inline
bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result)
{
    //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
    if (result == cudaSuccess)
        return true;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
    abort();
    return false;
}

#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

#endif // HeterogeneousCore_CUDAUtilities_cudaCheck_h

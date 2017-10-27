// includes, kernels
#include "sharedmem.cuh"


////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

template<class T>
__device__ void testKernel(T *g_idata, T *g_odata)
{
    // Shared mem size is determined by the host app at run time
    SharedMemory<T> smem;

    T *sdata = smem.getPointer();


    // access thread id
    const unsigned int tid = threadIdx.x;

    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];

    __syncthreads();


    // perform some computations
    sdata[tid] = (T) num_threads * sdata[tid];

    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}

extern "C" __global__ void testFloat(float *p1, float *p2) {  testKernel<float>(p1, p2); }

extern "C" __global__ void testInt(int *p1, int *p2) {  testKernel<int>(p1, p2); }

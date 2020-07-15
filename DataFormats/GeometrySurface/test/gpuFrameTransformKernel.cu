#include <cstdint>
#include <iostream>
#include <iomanip>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"

__global__ void toGlobal(SOAFrame<float> const* frame,
                         float const* xl,
                         float const* yl,
                         float* x,
                         float* y,
                         float* z,
                         float const* le,
                         float* ge,
                         uint32_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n)
    return;

  frame[0].toGlobal(xl[i], yl[i], x[i], y[i], z[i]);
  frame[0].toGlobal(le[3 * i], le[3 * i + 1], le[3 * i + 2], ge + 6 * i);
}

void toGlobalWrapper(SOAFrame<float> const* frame,
                     float const* xl,
                     float const* yl,
                     float* x,
                     float* y,
                     float* z,
                     float const* le,
                     float* ge,
                     uint32_t n) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CUDA toGlobal kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads"
            << std::endl;

  cms::cuda::launch(toGlobal, {blocksPerGrid, threadsPerBlock}, frame, xl, yl, x, y, z, le, ge, n);
}

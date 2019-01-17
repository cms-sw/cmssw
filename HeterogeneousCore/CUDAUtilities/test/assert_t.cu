#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

__global__
void testIt(int c){
  assert(c==1);
}

int main(int c, char **) {
  exitSansCUDADevices();

  testIt<<<1,1>>>(c);
  cudaDeviceSynchronize();
  return c==1;
}

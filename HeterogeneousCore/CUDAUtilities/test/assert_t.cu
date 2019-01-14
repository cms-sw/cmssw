#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

__global__
void testIt(int c){
  assert(c==1);
}

int main(int c, char **) {

  testIt<<<1,1>>>(c);
  cudaDeviceSynchronize();
  return c==1;
  
}

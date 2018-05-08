#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include <cstdint>


__global__
void toGlobal(SOAFrame<float> const * frame, 
	      float const * xl, float const * yl,
	      float * x, float * y, float * z,
	      uint32_t n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  
  frame[0].toGlobal(xl[i],yl[i],x[i],y[i],z[i]);
  
  
}

#include<iostream>
#include <iomanip>
#include "cuda/api_wrappers.h"


void toGlobalWrapper(SOAFrame<float> const * frame, 
	      float const * xl, float const * yl,
	      float * x, float * y, float * z,
		     uint32_t n) {

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  std::cout
    << "CUDA toGlobal kernel launch with " << blocksPerGrid
    << " blocks of " << threadsPerBlock << " threads" << std::endl;
 
  cuda::launch(
	       toGlobal,
	       { blocksPerGrid, threadsPerBlock },
	       frame, xl, yl, x, y, z,
	       n
	       );

}


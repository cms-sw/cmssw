/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg <E.Rozenberg@cwi.nl>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 *  Modified by VinInn for testing math funcs
 */

/* to run test
foreach f ( $CMSSW_BASE/test/$SCRAM_ARCH/DFM_Vector* )
echo $f; $f
end
*/

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "cuda/api_wrappers.h"

#include<DataFormats/Math/interface/approx_log.h>
#include<DataFormats/Math/interface/approx_exp.h>
#include<DataFormats/Math/interface/approx_atan2.h>

#ifdef __CUDACC__
#define inline __host__ __device__ inline
#include <vdt/sin.h>
#undef inline
#else
#include <vdt/sin.h>
#endif

std::mt19937 eng;
std::mt19937 eng2;
std::uniform_real_distribution<float> rgen(0.,1.);

constexpr float myExp(float x) {
  return  unsafe_expf<6>(x);
}

constexpr float myLog(float x) {
  return  unsafe_logf<6>(x);
}

__host__ __device__
inline float mySin(float x) {
  return vdt::fast_sinf(x);
}

constexpr int USEEXP=0, USESIN=1, USELOG=2;

template<int USE, bool ADDY=false>
// __host__ __device__
constexpr float testFunc(float x, float y) {
float ret=0;
if(USE==USEEXP)
  ret = myExp(x);
else if(USE==USESIN)
  ret = mySin(x);
else
  ret = myLog(x);
return ADDY ? ret+y : ret;
}

template<int USE, bool ADDY>
__global__ void vectorOp(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
  {
    C[i] = testFunc<USE,ADDY>(A[i],B[i]);
  }
}

template<int USE, bool ADDY>
void vectorOpH(const float *A, const float *B, float *C, int numElements)
{
  for (int i=0; i<numElements; ++i)
  {
    C[i] = testFunc<USE,ADDY>(A[i],B[i]);
  }
}

template<int USE, bool ADDY=false>
void go()
{
  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  auto current_device = cuda::device::current::get(); 

  int numElements = 200000;
  size_t size = numElements * sizeof(float);
  std::cout << "[Vector of " << numElements << " elements]\n";

  auto h_A = std::make_unique<float[]>(numElements);
  auto h_B = std::make_unique<float[]>(numElements);
  auto h_C = std::make_unique<float[]>(numElements);
  auto h_C2 = std::make_unique<float[]>(numElements);

  std::generate(h_A.get(), h_A.get() + numElements, [&](){return rgen(eng);});
  std::generate(h_B.get(), h_B.get() + numElements, [&](){return rgen(eng);});

  delta -= (std::chrono::high_resolution_clock::now()-start);
  auto d_A = cuda::memory::device::make_unique<float[]>(current_device, numElements);
  auto d_B = cuda::memory::device::make_unique<float[]>(current_device, numElements);
  auto d_C = cuda::memory::device::make_unique<float[]>(current_device, numElements);

  cuda::memory::copy(d_A.get(), h_A.get(), size);
  cuda::memory::copy(d_B.get(), h_B.get(), size);
  delta += (std::chrono::high_resolution_clock::now()-start);
  std::cout <<"cuda alloc+copy took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
    << " ms" << std::endl;

  // Launch the Vector OP CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  std::cout
    << "CUDA kernel launch with " << blocksPerGrid
    << " blocks of " << threadsPerBlock << " threads\n";

  delta -= (std::chrono::high_resolution_clock::now()-start);
  cuda::launch(
    vectorOp<USE,ADDY>,
    { blocksPerGrid, threadsPerBlock },
    d_A.get(), d_B.get(), d_C.get(), numElements
  );
  delta += (std::chrono::high_resolution_clock::now()-start);
  std::cout <<"cuda computation took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
    << " ms" << std::endl;

  delta -= (std::chrono::high_resolution_clock::now()-start);
  cuda::launch(
      vectorOp<USE,ADDY>,
      { blocksPerGrid, threadsPerBlock },
      d_A.get(), d_B.get(), d_C.get(), numElements
  );
  delta += (std::chrono::high_resolution_clock::now()-start);
  std::cout <<"cuda computation took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
    << " ms" << std::endl;

  delta -= (std::chrono::high_resolution_clock::now()-start);
  cuda::memory::copy(h_C.get(), d_C.get(), size);
  delta += (std::chrono::high_resolution_clock::now()-start);
  std::cout <<"cuda copy back took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
    << " ms" << std::endl;

  // on host now...
  delta -= (std::chrono::high_resolution_clock::now()-start);
  vectorOpH<USE,ADDY>(h_A.get(),h_B.get(),h_C2.get(),numElements);        
  delta += (std::chrono::high_resolution_clock::now()-start);
  std::cout <<"host computation took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
    << " ms" << std::endl;

  delta -= (std::chrono::high_resolution_clock::now()-start);
  vectorOpH<USE,ADDY>(h_A.get(),h_B.get(),h_C2.get(),numElements);
  delta += (std::chrono::high_resolution_clock::now()-start);
  std::cout <<"host computation took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
    << " ms" << std::endl;

  // Verify that the result vector is correct
  double ave = 0;
  int maxDiff = 0;
  long long ndiff=0;
  double fave = 0;
  float fmaxDiff = 0;
  for (int i = 0; i < numElements; ++i) {
    approx_math::binary32 g,c;
    g.f = testFunc<USE,ADDY>(h_A[i],h_B[i]);
    c.f = h_C[i];
    auto diff = std::abs(g.i32-c.i32);
    maxDiff = std::max(diff,maxDiff);
    ave += diff;
    if (diff!=0) ++ndiff;
    auto fdiff = std::abs(g.f-c.f);
    fave += fdiff;
    fmaxDiff = std::max(fdiff,fmaxDiff);
    //           if (diff>7)
    //           std::cerr << "Large diff at element " << i << ' ' << diff << ' ' << std::hexfloat 
    //                                  << g.f << "!=" << c.f << "\n";
  }
  std::cout << "ndiff ave, max " << ndiff << ' ' << ave/numElements << ' ' << maxDiff << std::endl;
  std::cout << "float ave, max " << fave/numElements << ' ' << fmaxDiff << std::endl;
  if (! ndiff) {
    std::cout << "Test PASSED\n";
    std::cout << "SUCCESS"<< std::endl;
  }
  cudaDeviceSynchronize();
}

int main() {
  int count = 0;
  auto status = cudaGetDeviceCount(& count);
  if (status != cudaSuccess) {
    std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
  }
  if (count == 0) {
    std::cerr << "No CUDA devices on this system, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
  }

  try {
    go<USEEXP>();
    go<USESIN>();
    go<USELOG>();

    go<USELOG, true>();
  } catch(cuda::runtime_error ex) {
    std::cerr << "CUDA error: " << ex.what() << std::endl;
    exit(EXIT_FAILURE);
  } catch(...) {
    std::cerr << "A non-CUDA error occurred" << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

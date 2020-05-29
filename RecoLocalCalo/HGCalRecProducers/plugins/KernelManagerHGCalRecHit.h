#ifndef RecoLocalCalo_HGCalRecProducers_KernelManager_HGCalRecHit_h
#define RecoLocalCalo_HGCalRecProducers_KernelManager_HGCalRecHit_h

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HGCalRecHitKernelImpl.cuh"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
//#include "Types.h"

#include <vector>
#include <algorithm> //std::swap  
#include <variant>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
extern __constant__ uint32_t calo_rechit_masks[];
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace {
  dim3 nblocks_;
  constexpr dim3 nthreads_(256); //some kernels will potentially not allocate shared memory properly with a lower number
}

template <typename T>
class KernelConstantData {
 public:
 KernelConstantData(T& data, HGCConstantVectorData& vdata): data_(data), vdata_(vdata) {
    if( ! (std::is_same<T, HGCeeUncalibratedRecHitConstantData>::value or std::is_same<T, HGChefUncalibratedRecHitConstantData>::value or std::is_same<T, HGChebUncalibratedRecHitConstantData>::value ))
      cms::cuda::LogError("WrongTemplateType") << "The KernelConstantData class does not support this type.";
  }
  T data_;
  HGCConstantVectorData vdata_;
};

template <typename TYPE_IN, typename TYPE_OUT>
  class KernelModifiableData {
 public:
 KernelModifiableData(int nhits, int stride, TYPE_IN *h_in, TYPE_IN *d_1, TYPE_IN *d_2, TYPE_OUT *d_out, TYPE_OUT *h_out):
  nhits_(nhits), stride_(stride), h_in_(h_in), d_1_(d_1), d_2_(d_2), d_out_(d_out), h_out_(h_out) {}

  int nhits_; //number of hits in the input event collection being processed
  int stride_; //modified number of hits so that warp (32 threads) boundary alignment is guaranteed
  TYPE_IN *h_in_; //host input data SoA
  TYPE_IN *d_1_, *d_2_; //device SoAs that handle all the processing steps applied to the input data. The pointers may be reused (ans swapped)
  TYPE_OUT *d_out_; //device SoA that stores the conversion of the hits to the new collection format
  TYPE_OUT *h_out_; //host SoA which receives the converted output collection from the GPU
};

class KernelManagerHGCalRecHit {
 public:
  KernelManagerHGCalRecHit(KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA>*);
  ~KernelManagerHGCalRecHit();
  void run_kernels(const KernelConstantData<HGCeeUncalibratedRecHitConstantData>*);
  void run_kernels(const KernelConstantData<HGChefUncalibratedRecHitConstantData>*, const hgcal_conditions::HeterogeneousHEFConditionsESProduct*);
  void run_kernels(const KernelConstantData<HGChebUncalibratedRecHitConstantData>*);
  HGCRecHitSoA* get_output();

 private:
  void after_();
  void transfer_soas_to_device_();
  void transfer_soa_to_host_and_synchronize_();
  void reuse_device_pointers_();

  int nbytes_host_;
  int nbytes_device_;
  KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA> *data_;
};

#endif //RecoLocalCalo_HGCalRecProducers_KernelManager_HGCalRecHit_h

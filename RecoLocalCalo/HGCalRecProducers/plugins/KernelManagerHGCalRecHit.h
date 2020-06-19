#ifndef RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h
#define RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitKernelImpl.cuh"
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

namespace { //kernel parameters
  dim3 nb_rechits_;
  constexpr dim3 nt_rechits_(256);
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
 KernelModifiableData(TYPE_IN *h_in, TYPE_IN *d_1, TYPE_IN *d_2, TYPE_OUT *d_out, TYPE_OUT *h_out):
  nhits_(0), stride_(0), h_in_(h_in), d_1_(d_1), d_2_(d_2), d_out_(d_out), h_out_(h_out) {}

  unsigned nhits_; //number of hits in the input event collection being processed
  unsigned stride_; //modified number of hits so that warp (32 threads) boundary alignment is guaranteed
  TYPE_IN *h_in_; //host input data SoA
  TYPE_IN *d_1_, *d_2_; //device SoAs that handle all the processing steps applied to the input data. The pointers may be reused (ans swapped)
  TYPE_OUT *d_out_; //device SoA that stores the conversion of the hits to the new collection format
  TYPE_OUT *h_out_; //host SoA which receives the converted output collection from the GPU
};

class KernelManagerHGCalRecHit {
 public:
  KernelManagerHGCalRecHit();
  KernelManagerHGCalRecHit(KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA>*);
  ~KernelManagerHGCalRecHit();
  void run_kernels(const KernelConstantData<HGCeeUncalibratedRecHitConstantData>*, const cudaStream_t&);
  void run_kernels(const KernelConstantData<HGChefUncalibratedRecHitConstantData>*, const cudaStream_t&);
  void run_kernels(const KernelConstantData<HGChebUncalibratedRecHitConstantData>*, const cudaStream_t&);
  HGCRecHitSoA* get_output();

 private:
  void transfer_soas_to_device_(const cudaStream_t&);
  void transfer_soa_to_host_and_synchronize_(const cudaStream_t&);
  void reuse_device_pointers_();

  int nbytes_host_;
  int nbytes_device_;
  KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA> *data_;
};

#endif //RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h

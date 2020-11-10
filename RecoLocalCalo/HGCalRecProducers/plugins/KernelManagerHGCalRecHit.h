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
  constexpr dim3 nt_rechits_(1024);
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
  void transfer_soa_to_host_(const cudaStream_t&);
  void reuse_device_pointers_();

  int nbytes_host_;
  int nbytes_device_;
  KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA> *data_;
};

#endif //RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h

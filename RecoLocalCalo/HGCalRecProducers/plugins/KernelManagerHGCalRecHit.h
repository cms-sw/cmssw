#ifndef RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h
#define RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitKernelImpl.cuh"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"

#include <vector>
#include <algorithm>  //std::swap
#include <variant>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
extern __constant__ uint32_t calo_rechit_masks[];
#endif

template <typename T>
class KernelConstantData {
public:
  KernelConstantData(T& data, HGCConstantVectorData& vdata) : data_(data), vdata_(vdata) {
    static_assert(std::is_same<T, HGCeeUncalibRecHitConstantData>::value or
                      std::is_same<T, HGChefUncalibRecHitConstantData>::value or
                      std::is_same<T, HGChebUncalibRecHitConstantData>::value,
                  "The KernelConstantData class does not support this type.");
  }
  T data_;
  HGCConstantVectorData vdata_;
};

class KernelManagerHGCalRecHit {
public:
  KernelManagerHGCalRecHit();
  KernelManagerHGCalRecHit(const HGCUncalibRecHitSoA&, const HGCUncalibRecHitSoA&, const HGCRecHitSoA&);
  KernelManagerHGCalRecHit(const HGCRecHitSoA&, const ConstHGCRecHitSoA&);
  ~KernelManagerHGCalRecHit();
  void run_kernels(const KernelConstantData<HGCeeUncalibRecHitConstantData>*, const cudaStream_t&);
  void run_kernels(const KernelConstantData<HGChefUncalibRecHitConstantData>*, const cudaStream_t&);
  void run_kernels(const KernelConstantData<HGChebUncalibRecHitConstantData>*, const cudaStream_t&);
  void transfer_soa_to_host(const cudaStream_t&);

private:
  void transfer_soa_to_device_(const cudaStream_t&);

  uint32_t nhits_;
  uint32_t pad_;
  uint32_t nbytes_host_;
  uint32_t nbytes_device_;
  HGCUncalibRecHitSoA h_uncalibSoA_, d_uncalibSoA_;
  HGCRecHitSoA h_calibSoA_, d_calibSoA_;
  ConstHGCRecHitSoA d_calibConstSoA_;
};

#endif  //RecoLocalCalo_HGCalRecProducers_KernelManagerHGCalRecHit_h

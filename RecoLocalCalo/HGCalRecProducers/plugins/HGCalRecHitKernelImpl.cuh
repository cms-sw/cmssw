#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitKernelImpl_cuh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitKernelImpl_cuh

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitsToRecHitsConstants.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

__global__ void ee_to_rechit(HGCRecHitSoA dst_soa,
                             HGCUncalibRecHitSoA src_soa,
                             HGCeeUncalibRecHitConstantData cdata,
                             int length);

__global__ void hef_to_rechit(HGCRecHitSoA dst_soa,
                              HGCUncalibRecHitSoA src_soa,
                              HGChefUncalibRecHitConstantData cdata,
                              int length);

__global__ void heb_to_rechit(HGCRecHitSoA dst_soa,
                              HGCUncalibRecHitSoA src_soa,
                              HGChebUncalibRecHitConstantData cdata,
                              int length);

#endif  //RecoLocalCalo_HGCalRecProducers_HGCalRecHitKernelImpl_cuh

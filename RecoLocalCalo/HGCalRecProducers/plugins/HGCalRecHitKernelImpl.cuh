#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitKernelImpl_cuh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitKernelImpl_cuh

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataFormats/HGCal/interface/HGCUncalibratedRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibratedRecHitsToRecHitsConstants.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

__global__
void ee_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGCeeUncalibratedRecHitConstantData cdata, int length);

__global__
void hef_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChefUncalibratedRecHitConstantData cdata, int length);

__global__
void heb_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChebUncalibratedRecHitConstantData cdata, int length);

__global__
void ee_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGCeeUncalibratedRecHitConstantData cdata, int length);

__global__
void hef_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChefUncalibratedRecHitConstantData cdata, const hgcal_conditions::HeterogeneousHEFConditionsESProduct*, int length);

__global__
void heb_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChebUncalibratedRecHitConstantData cdata, int length);

__global__
void fill_positions_from_detids(const hgcal_conditions::HeterogeneousHEFConditionsESProduct* conds);
  
__global__
void print_positions_from_detids(const hgcal_conditions::HeterogeneousHEFConditionsESProduct* conds);
  
#endif //RecoLocalCalo_HGCalRecProducers_HGCalRecHitKernelImpl_cuh

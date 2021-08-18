#ifndef RecoLocalCalo_HGCalESProducers_HGCalCellPositionsKernelImpl_cuh
#define RecoLocalCalo_HGCalESProducers_HGCalCellPositionsKernelImpl_cuh

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"

__global__ void fill_positions_from_detids(
    const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* conds);

__global__ void print_positions_from_detids(
    const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* conds);

__global__ void test(uint32_t detid_test,
                     const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* conds);

#endif  //RecoLocalCalo_HGCalESProducers_HGCalCellPositionsKernelImpl_cuh

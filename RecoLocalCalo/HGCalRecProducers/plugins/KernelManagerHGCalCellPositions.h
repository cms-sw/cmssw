#ifndef RecoLocalCalo_HGCalESProducers_KernelManagerHGCalCellPositions_h
#define RecoLocalCalo_HGCalESProducers_KernelManagerHGCalCellPositions_h

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

#include <vector>
#include <algorithm>  //std::swap
#include <variant>
#include <cuda.h>
#include <cuda_runtime.h>

/*
#ifdef __CUDA_ARCH__
extern __constant__ uint32_t calo_rechit_masks[];
#endif
*/

class KernelManagerHGCalCellPositions {
public:
  KernelManagerHGCalCellPositions(const size_t&);

  void fill_positions(const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct*);
  void test_cell_positions(unsigned, const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct*);
};

#endif  //RecoLocalCalo_HGCalESProducers_KernelManagerHGCalCellPositions_h

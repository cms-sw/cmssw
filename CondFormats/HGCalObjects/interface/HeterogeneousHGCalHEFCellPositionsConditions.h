#ifndef CondFormats_HGCalObjects_HeterogeneousHGCalHEFConditions_h
#define CondFormats_HGCalObjects_HeterogeneousHGCalHEFConditions_h

#include <numeric>  //accumulate
#include <typeinfo>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"

namespace cpar = hgcal_conditions::parameters;
namespace cpos = hgcal_conditions::positions;

// Declare the wrapper ESProduct. The corresponding ESProducer should
// produce objects of this type.
class HeterogeneousHGCalHEFCellPositionsConditions {
public:
  // Constructor takes the standard CPU ESProduct, and transforms the
  // necessary data to array(s) in pinned host memory
  HeterogeneousHGCalHEFCellPositionsConditions(cpos::HGCalPositionsMapping*);

  // Deallocates all pinned host memory
  ~HeterogeneousHGCalHEFCellPositionsConditions();

  // Function to return the actual payload on the memory of the current device
  hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct const* getHeterogeneousConditionsESProductAsync(
      cudaStream_t stream) const;

private:
  // Holds the data in pinned CPU memory
  // Contrary to its non-heterogeneous counterpart (constructor argument) it is *not* a pointer (so to avoid an extra allocation)
  cpos::HeterogeneousHGCalPositionsMapping posmap_;
  size_t nelems_posmap_;

  std::vector<size_t> sizes_;
  size_t chunk_;
  const size_t number_position_arrays =
      2;  //x and y; required due to the assymetry between cpos::HeterogeneousHGCalPositionsMapping and cpos::HGCalPositionsMapping

  std::vector<size_t> calculate_memory_bytes_(cpos::HGCalPositionsMapping*);
  size_t allocate_memory_(const std::vector<size_t>&);
  void transfer_data_to_heterogeneous_pointers_(const std::vector<size_t>&, cpos::HGCalPositionsMapping*);
  void transfer_data_to_heterogeneous_vars_(const cpos::HGCalPositionsMapping*);

  /*methods for managing SoA's pointers*/
  //float
  float*& select_pointer_f_(cpos::HeterogeneousHGCalPositionsMapping*, const unsigned int&) const;
  std::vector<float>& select_pointer_f_(cpos::HGCalPositionsMapping*, const unsigned int&);
  //int32_t
  int32_t*& select_pointer_i_(cpos::HeterogeneousHGCalPositionsMapping*, const unsigned int&) const;
  std::vector<int32_t>& select_pointer_i_(cpos::HGCalPositionsMapping*, const unsigned int&);
  //uint32_t
  uint32_t*& select_pointer_u_(cpos::HeterogeneousHGCalPositionsMapping*, const unsigned int&) const;
  std::vector<uint32_t>& select_pointer_u_(cpos::HGCalPositionsMapping*, const unsigned int&);

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    // internal pointers are on device, struct itself is on CPU
    hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* host = nullptr;
    // internal pounters and struct are on device
    hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* device = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif  //CondFormats_HGCalObjects_HeterogeneousHGCalHEFConditions_h

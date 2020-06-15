#ifndef RecoLocalCalo_HGCRecProducers_HeterogeneousHGCalHEFConditions_h
#define RecoLocalCalo_HGCRecProducers_HeterogeneousHGCalHEFConditions_h

#include <numeric> //accumulate
#include <typeinfo>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

namespace cpar = hgcal_conditions::parameters;
namespace cpos = hgcal_conditions::positions;  

// Declare the wrapper ESProduct. The corresponding ESProducer should
// produce objects of this type.
class HeterogeneousHGCalHEFConditionsWrapper {
 public:
  // Constructor takes the standard CPU ESProduct, and transforms the
  // necessary data to array(s) in pinned host memory
  HeterogeneousHGCalHEFConditionsWrapper(const HGCalParameters*, cpos::HGCalPositionsMapping*);
  
  // Deallocates all pinned host memory
  ~HeterogeneousHGCalHEFConditionsWrapper();
  
  // Function to return the actual payload on the memory of the current device
  hgcal_conditions::HeterogeneousHEFConditionsESProduct const *getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const;

 private:
  // Holds the data in pinned CPU memory
  // Contrary to its non-heterogeneous counterpart (constructor argument) it is *not* a pointer (so to avoid an extra allocation)
  cpar::HeterogeneousHGCalHEFParameters params_;
  cpos::HeterogeneousHGCalPositionsMapping posmap_;

  std::vector<size_t> sizes_params_;
  std::vector<size_t> sizes_pos_;
  size_t chunk_params_;
  size_t chunk_pos_;

  std::vector<size_t> calculate_memory_bytes_params_(const HGCalParameters*);
  std::vector<size_t> calculate_memory_bytes_pos_(cpos::HGCalPositionsMapping*);
  size_t allocate_memory_params_(const std::vector<size_t>&);
  size_t allocate_memory_pos_(const std::vector<size_t>&);
  void transfer_data_to_heterogeneous_pointers_params_(const std::vector<size_t>&, const HGCalParameters*);
  void transfer_data_to_heterogeneous_pointers_pos_(const std::vector<size_t>&, cpos::HGCalPositionsMapping*);
  void transfer_data_to_heterogeneous_vars_pos_(const cpos::HGCalPositionsMapping*);

  /*methods for managing SoA's pointers*/
  //double
  double*& select_pointer_d_(cpar::HeterogeneousHGCalHEFParameters*, const unsigned int&) const;
  std::vector<double> select_pointer_d_(const HGCalParameters*, const unsigned int&) const;
  //int32_t
  int32_t*& select_pointer_i_(cpar::HeterogeneousHGCalHEFParameters*, const unsigned int&) const;
  std::vector<int32_t> select_pointer_i_(const HGCalParameters*, const unsigned int&) const;
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
    hgcal_conditions::HeterogeneousHEFConditionsESProduct *host = nullptr;
    // internal pounters and struct are on device
    hgcal_conditions::HeterogeneousHEFConditionsESProduct *device = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif //RecoLocalCalo_HGCRecProducers_HeterogeneousHGCalHEFConditions_h


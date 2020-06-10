#ifndef HeterogeneousHGCalHEFConditions_h
#define HeterogeneousHGCalHEFConditions_h

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
  HeterogeneousHGCalHEFConditionsWrapper(const HGCalParameters*, const cpos::HGCalPositions*);
  
  // Deallocates all pinned host memory
  ~HeterogeneousHGCalHEFConditionsWrapper();
  
  // Function to return the actual payload on the memory of the current device
  hgcal_conditions::HeterogeneousHEFConditionsESProduct const *getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const;

 private:
  // Holds the data in pinned CPU memory
  // Contrary to its non-heterogeneous counterpart (constructor argument) it is *not* a pointer (so to avoid an extra allocation)
  cpar::HeterogeneousHGCalHEFParameters params_;
  cpos::HeterogeneousHGCalPositions pos_;

  std::vector<size_t> sizes_params_;
  std::vector<size_t> sizes_pos_;
  size_t chunk_params_;
  size_t chunk_pos_;

  std::vector<size_t> calculate_memory_bytes_params_(const HGCalParameters*);
  std::vector<size_t> calculate_memory_bytes_pos_(const cpos::HGCalPositions*);
  size_t allocate_memory_params_(const std::vector<size_t>&);
  size_t allocate_memory_pos_(const std::vector<size_t>&);
  void transfer_data_to_heterogeneous_pointers_params_(const std::vector<size_t>&, const HGCalParameters*);
  void transfer_data_to_heterogeneous_pointers_pos_(const std::vector<size_t>&, const cpos::HGCalPositions*);
  
  double*& select_pointer_d(cpar::HeterogeneousHGCalHEFParameters*, const unsigned int&) const;
  std::vector<double> select_pointer_d(const HGCalParameters*, const unsigned int&) const;
  int32_t*& select_pointer_i(cpar::HeterogeneousHGCalHEFParameters*, const unsigned int&) const;
  std::vector<int32_t> select_pointer_i(const HGCalParameters*, const unsigned int&) const;
  float*& select_pointer_f(cpos::HeterogeneousHGCalPositions*, const unsigned int&) const;
  std::vector<float> select_pointer_f(const cpos::HGCalPositions*, const unsigned int&) const;

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

#endif //HeterogeneousHGCalHEFConditions_h


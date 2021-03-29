#ifndef HeterogeneousHGCalEEConditions_h
#define HeterogeneousHGCalEEConditions_h

#include <numeric>  //accumulate
#include <typeinfo>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

namespace cp = hgcal_conditions::parameters;

// Declare the wrapper ESProduct. The corresponding ESProducer should
// produce objects of this type.
class HeterogeneousHGCalEEConditionsWrapper {
public:
  // Constructor takes the standard CPU ESProduct, and transforms the
  // necessary data to array(s) in pinned host memory
  HeterogeneousHGCalEEConditionsWrapper(const HGCalParameters *);

  // Deallocates all pinned host memory
  ~HeterogeneousHGCalEEConditionsWrapper();

  // Function to return the actual payload on the memory of the current device
  hgcal_conditions::HeterogeneousEEConditionsESProduct const *getHeterogeneousConditionsESProductAsync(
      cudaStream_t stream) const;

private:
  // Holds the data in pinned CPU memory
  // Contrary to its non-heterogeneous counterpart (constructor argument) it is *not* a pointer (so to avoid an extra allocation)
  cp::HeterogeneousHGCalEEParameters params_;

  std::vector<size_t> sizes_;
  size_t chunk_;

  void calculate_memory_bytes(const HGCalParameters *);

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    // internal pointers are on device, struct itself is on CPU
    hgcal_conditions::HeterogeneousEEConditionsESProduct *host = nullptr;
    // internal pounters and struct are on device
    hgcal_conditions::HeterogeneousEEConditionsESProduct *device = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif  //HeterogeneousHGCalEEConditions_h

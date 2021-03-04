#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFConditions.h"

HeterogeneousHGCalHEFConditionsWrapper::HeterogeneousHGCalHEFConditionsWrapper(
    const HGCalParameters* cpuHGCalParameters) {
  //HGCalParameters as defined in CMSSW
  this->sizes_params_ = calculate_memory_bytes_params_(cpuHGCalParameters);
  this->chunk_params_ = allocate_memory_params_(this->sizes_params_);
  transfer_data_to_heterogeneous_pointers_params_(this->sizes_params_, cpuHGCalParameters);
}

size_t HeterogeneousHGCalHEFConditionsWrapper::allocate_memory_params_(const std::vector<size_t>& sz) {
  size_t chunk_ = std::accumulate(sz.begin(), sz.end(), 0);  //total memory required in bytes
  cudaCheck(cudaMallocHost(&this->params_.cellFineX_, chunk_));
  return chunk_;
}

void HeterogeneousHGCalHEFConditionsWrapper::transfer_data_to_heterogeneous_pointers_params_(
    const std::vector<size_t>& sz, const HGCalParameters* cpuParams) {
  //store cumulative sum in bytes and convert it to sizes in units of C++ typesHEF, i.e., number if items to be transferred to GPU
  std::vector<size_t> cumsum_sizes(sz.size() + 1, 0);  //starting with zero
  std::partial_sum(sz.begin(), sz.end(), cumsum_sizes.begin() + 1);
  for (unsigned int i = 1; i < cumsum_sizes.size(); ++i)  //start at second element (the first is zero)
  {
    size_t typesHEFsize = 0;
    if (cpar::typesHEF[i - 1] == cpar::HeterogeneousHGCalHEFParametersType::Double)
      typesHEFsize = sizeof(double);
    else if (cpar::typesHEF[i - 1] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t)
      typesHEFsize = sizeof(int32_t);
    else
      throw cms::Exception("HeterogeneousHGCalHEFConditionsWrapper") << "Wrong HeterogeneousHGCalParameters type";
    cumsum_sizes[i] /= typesHEFsize;
  }

  for (unsigned int j = 0; j < sz.size(); ++j) {
    //setting the pointers
    if (j != 0) {
      const unsigned int jm1 = j - 1;
      const size_t shift = cumsum_sizes[j] - cumsum_sizes[jm1];
      if (cpar::typesHEF[jm1] == cpar::HeterogeneousHGCalHEFParametersType::Double and
          cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double)
        select_pointer_d_(&this->params_, j) = select_pointer_d_(&this->params_, jm1) + shift;
      else if (cpar::typesHEF[jm1] == cpar::HeterogeneousHGCalHEFParametersType::Double and
               cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t)
        select_pointer_i_(&this->params_, j) =
            reinterpret_cast<int32_t*>(select_pointer_d_(&this->params_, jm1) + shift);
    }

    //copying the pointers' content
    for (unsigned int i = cumsum_sizes[j]; i < cumsum_sizes[j + 1]; ++i) {
      unsigned int index = i - cumsum_sizes[j];
      if (cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double) {
        select_pointer_d_(&this->params_, j)[index] = select_pointer_d_(cpuParams, j)[index];
      } else if (cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t) {
        select_pointer_i_(&this->params_, j)[index] = select_pointer_i_(cpuParams, j)[index];
      } else
        throw cms::Exception("HeterogeneousHGCalHEFConditionsWrapper") << "Wrong HeterogeneousHGCalParameters type";
    }
  }
}

std::vector<size_t> HeterogeneousHGCalHEFConditionsWrapper::calculate_memory_bytes_params_(
    const HGCalParameters* cpuParams) {
  size_t npointers = hgcal_conditions::parameters::typesHEF.size();
  std::vector<size_t> sizes(npointers);
  for (unsigned int i = 0; i < npointers; ++i) {
    if (cpar::typesHEF[i] == cpar::HeterogeneousHGCalHEFParametersType::Double)
      sizes[i] = select_pointer_d_(cpuParams, i).size();
    else
      sizes[i] = select_pointer_i_(cpuParams, i).size();
  }

  std::vector<size_t> sizes_units(npointers);
  for (unsigned int i = 0; i < npointers; ++i) {
    if (cpar::typesHEF[i] == cpar::HeterogeneousHGCalHEFParametersType::Double)
      sizes_units[i] = sizeof(double);
    else if (cpar::typesHEF[i] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t)
      sizes_units[i] = sizeof(int32_t);
  }

  //element by element multiplication
  this->sizes_params_.resize(npointers);
  std::transform(
      sizes.begin(), sizes.end(), sizes_units.begin(), this->sizes_params_.begin(), std::multiplies<size_t>());
  return this->sizes_params_;
}

HeterogeneousHGCalHEFConditionsWrapper::~HeterogeneousHGCalHEFConditionsWrapper() {
  cudaCheck(cudaFreeHost(this->params_.cellFineX_));
}

//I could use template specializations
//try to use std::variant in the future to avoid similar functions with different return values
double*& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_d_(cpar::HeterogeneousHGCalHEFParameters* cpuObject,
                                                                   const unsigned int& item) const {
  switch (item) {
    case 0:
      return cpuObject->cellFineX_;
    case 1:
      return cpuObject->cellFineY_;
    case 2:
      return cpuObject->cellCoarseX_;
    case 3:
      return cpuObject->cellCoarseY_;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_d(heterogeneous): no item.";
      return cpuObject->cellCoarseY_;
  }
}

std::vector<double> HeterogeneousHGCalHEFConditionsWrapper::select_pointer_d_(const HGCalParameters* cpuObject,
                                                                              const unsigned int& item) const {
  switch (item) {
    case 0:
      return cpuObject->cellFineX_;
    case 1:
      return cpuObject->cellFineY_;
    case 2:
      return cpuObject->cellCoarseX_;
    case 3:
      return cpuObject->cellCoarseY_;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_d(non-heterogeneous): no item.";
      return cpuObject->cellCoarseY_;
  }
}

int32_t*& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_i_(cpar::HeterogeneousHGCalHEFParameters* cpuObject,
                                                                    const unsigned int& item) const {
  switch (item) {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_i(heterogeneous): no item.";
      return cpuObject->waferTypeL_;
  }
}

std::vector<int32_t> HeterogeneousHGCalHEFConditionsWrapper::select_pointer_i_(const HGCalParameters* cpuObject,
                                                                               const unsigned int& item) const {
  switch (item) {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_i(non-heterogeneous): no item.";
      return cpuObject->waferTypeL_;
  }
}

hgcal_conditions::HeterogeneousHEFConditionsESProduct const*
HeterogeneousHGCalHEFConditionsWrapper::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
  // cms::cuda::ESProduct<T> essentially holds an array of GPUData objects,
  // one per device. If the data have already been transferred to the
  // current device (or the transfer has been queued), the helper just
  // returns a reference to that GPUData object. Otherwise, i.e. data are
  // not yet on the current device, the helper calls the lambda to do the
  // necessary memory allocations and to queue the transfers.
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate the payload object on pinned host memory.
    cudaCheck(cudaMallocHost(&data.host, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct)));
    // Allocate the payload array(s) on device memory.
    cudaCheck(cudaMalloc(&(data.host->params.cellFineX_), chunk_params_));

    // Complete the host-side information on the payload

    //(set the pointers of the parameters)
    size_t sdouble = sizeof(double);
    for (unsigned int j = 0; j < this->sizes_params_.size() - 1; ++j) {
      if (cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double and
          cpar::typesHEF[j + 1] == cpar::HeterogeneousHGCalHEFParametersType::Double)
        select_pointer_d_(&(data.host->params), j + 1) =
            select_pointer_d_(&(data.host->params), j) + (this->sizes_params_[j] / sdouble);
      else if (cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double and
               cpar::typesHEF[j + 1] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t)
        select_pointer_i_(&(data.host->params), j + 1) =
            reinterpret_cast<int32_t*>(select_pointer_d_(&(data.host->params), j) + (this->sizes_params_[j] / sdouble));
      else
        throw cms::Exception("HeterogeneousHGCalHEFConditionsWrapper")
            << "compare this functions' logic with hgcal_conditions::parameters::typesHEF";
    }

    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.device, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct)));
    // Transfer the payload, first the array(s) ...
    cudaCheck(cudaMemcpyAsync(
        data.host->params.cellFineX_, this->params_.cellFineX_, chunk_params_, cudaMemcpyHostToDevice, stream));

    // ... and then the payload object
    cudaCheck(cudaMemcpyAsync(data.device,
                              data.host,
                              sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct),
                              cudaMemcpyHostToDevice,
                              stream));
  });  //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousHGCalHEFConditionsWrapper::GPUData::~GPUData() {
  if (host != nullptr) {
    cudaCheck(cudaFree(host->params.cellFineX_));
    cudaCheck(cudaFreeHost(host));
  }
  cudaCheck(cudaFree(device));
}

#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEBConditions.h"

HeterogeneousHGCalHEBConditionsWrapper::HeterogeneousHGCalHEBConditionsWrapper(
    const HGCalParameters* cpuHGCalParameters) {
  calculate_memory_bytes(cpuHGCalParameters);

  chunk_ = std::accumulate(this->sizes_.begin(), this->sizes_.end(), 0);  //total memory required in bytes
  cudaCheck(cudaMallocHost(&this->params_.testD_, chunk_));

  //store cumulative sum in bytes and convert it to sizes in units of C++ typesHEB, i.e., number if items to be transferred to GPU
  std::vector<size_t> cumsum_sizes(this->sizes_.size() + 1, 0);  //starting with zero
  std::partial_sum(this->sizes_.begin(), this->sizes_.end(), cumsum_sizes.begin() + 1);
  for (unsigned int i = 1; i < cumsum_sizes.size(); ++i)  //start at second element (the first is zero)
  {
    unsigned int typesHEBize = 0;
    if (cp::typesHEB[i - 1] == cp::HeterogeneousHGCalHEBParametersType::Double)
      typesHEBize = sizeof(double);
    else if (cp::typesHEB[i - 1] == cp::HeterogeneousHGCalHEBParametersType::Int32_t)
      typesHEBize = sizeof(int32_t);
    else
      throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper") << "Wrong HeterogeneousHGCalParameters type";
    cumsum_sizes[i] /= typesHEBize;
  }

  for (unsigned int j = 0; j < this->sizes_.size(); ++j) {
    //setting the pointers
    if (j != 0) {
      const unsigned int jm1 = j - 1;
      if (cp::typesHEB[jm1] == cp::HeterogeneousHGCalHEBParametersType::Double and
          cp::typesHEB[j] == cp::HeterogeneousHGCalHEBParametersType::Double)
        select_pointer_d(&this->params_, j) = select_pointer_d(&this->params_, jm1) + this->sizes_[jm1];
      else if (cp::typesHEB[jm1] == cp::HeterogeneousHGCalHEBParametersType::Double and
               cp::typesHEB[j] == cp::HeterogeneousHGCalHEBParametersType::Int32_t)
        select_pointer_i(&this->params_, j) =
            reinterpret_cast<int32_t*>(select_pointer_d(&this->params_, jm1) + this->sizes_[jm1]);
    }

    //copying the pointers' content
    for (unsigned int i = cumsum_sizes[j]; i < cumsum_sizes[j + 1]; ++i) {
      unsigned int index = i - cumsum_sizes[j];
      if (cp::typesHEB[j] == cp::HeterogeneousHGCalHEBParametersType::Double) {
        select_pointer_d(&this->params_, j)[index] = select_pointer_d(cpuHGCalParameters, j)[index];
      } else if (cp::typesHEB[j] == cp::HeterogeneousHGCalHEBParametersType::Int32_t)
        select_pointer_i(&this->params_, j)[index] = select_pointer_i(cpuHGCalParameters, j)[index];
      else
        throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper") << "Wrong HeterogeneousHGCalParameters type";
    }
  }
}

void HeterogeneousHGCalHEBConditionsWrapper::calculate_memory_bytes(const HGCalParameters* cpuHGCalParameters) {
  size_t npointers = hgcal_conditions::parameters::typesHEB.size();
  std::vector<size_t> sizes(npointers);
  for (unsigned int i = 0; i < npointers; ++i) {
    if (cp::typesHEB[i] == cp::HeterogeneousHGCalHEBParametersType::Double)
      sizes[i] = select_pointer_d(cpuHGCalParameters, i).size();
    else
      sizes[i] = select_pointer_i(cpuHGCalParameters, i).size();
  }

  std::vector<size_t> sizes_units(npointers);
  for (unsigned int i = 0; i < npointers; ++i) {
    if (cp::typesHEB[i] == cp::HeterogeneousHGCalHEBParametersType::Double)
      sizes_units[i] = sizeof(double);
    else if (cp::typesHEB[i] == cp::HeterogeneousHGCalHEBParametersType::Int32_t)
      sizes_units[i] = sizeof(int32_t);
  }

  //element by element multiplication
  this->sizes_.resize(npointers);
  std::transform(sizes.begin(), sizes.end(), sizes_units.begin(), this->sizes_.begin(), std::multiplies<size_t>());
}

HeterogeneousHGCalHEBConditionsWrapper::~HeterogeneousHGCalHEBConditionsWrapper() {
  cudaCheck(cudaFreeHost(this->params_.testD_));
}

//I could use template specializations
//try to use std::variant in the future to avoid similar functions with different return values
double*& HeterogeneousHGCalHEBConditionsWrapper::select_pointer_d(cp::HeterogeneousHGCalHEBParameters* cpuObject,
                                                                  const unsigned int& item) const {
  switch (item) {
    case 0:
      return cpuObject->testD_;
    default:
      throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper") << "select_pointer_d(heterogeneous): no item.";
      return cpuObject->testD_;
  }
}

std::vector<double> HeterogeneousHGCalHEBConditionsWrapper::select_pointer_d(const HGCalParameters* cpuObject,
                                                                             const unsigned int& item) const {
  switch (item) {
    case 0:
      return cpuObject->cellFineX_;
    default:
      throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper") << "select_pointer_d(non-heterogeneous): no item.";
      return cpuObject->cellFineX_;
  }
}

int32_t*& HeterogeneousHGCalHEBConditionsWrapper::select_pointer_i(cp::HeterogeneousHGCalHEBParameters* cpuObject,
                                                                   const unsigned int& item) const {
  switch (item) {
    case 1:
      return cpuObject->testI_;
    default:
      throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper") << "select_pointer_i(heterogeneous): no item.";
      return cpuObject->testI_;
  }
}

std::vector<int32_t> HeterogeneousHGCalHEBConditionsWrapper::select_pointer_i(const HGCalParameters* cpuObject,
                                                                              const unsigned int& item) const {
  switch (item) {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper") << "select_pointer_i(non-heterogeneous): no item.";
      return cpuObject->waferTypeL_;
  }
}

hgcal_conditions::HeterogeneousHEBConditionsESProduct const*
HeterogeneousHGCalHEBConditionsWrapper::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
  // cms::cuda::ESProduct<T> essentially holds an array of GPUData objects,
  // one per device. If the data have already been transferred to the
  // current device (or the transfer has been queued), the helper just
  // returns a reference to that GPUData object. Otherwise, i.e. data are
  // not yet on the current device, the helper calls the lambda to do the
  // necessary memory allocations and to queue the transfers.
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate the payload object on pinned host memory.
    cudaCheck(cudaMallocHost(&data.host, sizeof(hgcal_conditions::HeterogeneousHEBConditionsESProduct)));
    // Allocate the payload array(s) on device memory.
    cudaCheck(cudaMalloc(&(data.host->params.testD_), chunk_));

    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.device, sizeof(hgcal_conditions::HeterogeneousHEBConditionsESProduct)));
    // Transfer the payload, first the array(s) ...
    cudaCheck(cudaMemcpyAsync(data.host->params.testD_, this->params_.testD_, chunk_, cudaMemcpyHostToDevice, stream));

    for (unsigned int j = 0; j < this->sizes_.size() - 1; ++j) {
      if (cp::typesHEB[j] == cp::HeterogeneousHGCalHEBParametersType::Double and
          cp::typesHEB[j + 1] == cp::HeterogeneousHGCalHEBParametersType::Double)
        select_pointer_d(&(data.host->params), j + 1) = select_pointer_d(&(data.host->params), j) + this->sizes_[j];
      else if (cp::typesHEB[j] == cp::HeterogeneousHGCalHEBParametersType::Double and
               cp::typesHEB[j + 1] == cp::HeterogeneousHGCalHEBParametersType::Int32_t)
        select_pointer_i(&(data.host->params), j + 1) =
            reinterpret_cast<int32_t*>(select_pointer_d(&(data.host->params), j) + this->sizes_[j]);
      else
        throw cms::Exception("HeterogeneousHGCalHEBConditionsWrapper")
            << "compare this functions' logic with hgcal_conditions::parameters::typesHEB";
    }

    // ... and then the payload object
    cudaCheck(cudaMemcpyAsync(data.device,
                              data.host,
                              sizeof(hgcal_conditions::HeterogeneousHEBConditionsESProduct),
                              cudaMemcpyHostToDevice,
                              stream));
  });  //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousHGCalHEBConditionsWrapper::GPUData::~GPUData() {
  if (host != nullptr) {
    cudaCheck(cudaFree(host->params.testD_));
    cudaCheck(cudaFreeHost(host));
  }
  cudaCheck(cudaFree(device));
}

//template double*& HeterogeneousHGCalHEBConditionsWrapper::select_pointer_d<cp::HeterogeneousHGCalParameters*>(cp::HeterogeneousHGCalParameters*, const unsigned int&) const;

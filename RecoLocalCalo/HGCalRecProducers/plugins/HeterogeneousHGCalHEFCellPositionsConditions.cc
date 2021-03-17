#include "CondFormats/HGCalObjects/interface/HeterogeneousHGCalHEFCellPositionsConditions.h"

HeterogeneousHGCalHEFCellPositionsConditions::HeterogeneousHGCalHEFCellPositionsConditions(
    cpos::HGCalPositionsMapping* cpuPos) {
  //HGCalPositions as defined in hgcal_conditions::positions
  this->sizes_ = calculate_memory_bytes_(cpuPos);
  this->chunk_ = allocate_memory_(this->sizes_);
  transfer_data_to_heterogeneous_pointers_(this->sizes_, cpuPos);
  transfer_data_to_heterogeneous_vars_(cpuPos);
}

size_t HeterogeneousHGCalHEFCellPositionsConditions::allocate_memory_(const std::vector<size_t>& sz) {
  size_t chunk = std::accumulate(sz.begin(), sz.end(), 0);  //total memory required in bytes
  cudaCheck(cudaMallocHost(&this->posmap_.x, chunk));
  return chunk;
}

void HeterogeneousHGCalHEFCellPositionsConditions::transfer_data_to_heterogeneous_pointers_(
    const std::vector<size_t>& sz, cpos::HGCalPositionsMapping* cpuPos) {
  //store cumulative sum in bytes and convert it to sizes in units of C++ typesHEF, i.e., number if items to be transferred to GPU
  std::vector<size_t> cumsum_sizes(sz.size() + 1, 0);  //starting with zero
  std::partial_sum(sz.begin(), sz.end(), cumsum_sizes.begin() + 1);
  for (unsigned int i = 1; i < cumsum_sizes.size(); ++i)  //start at second element (the first is zero)
  {
    size_t types_size = 0;
    if (cpos::types[i - 1] == cpos::HeterogeneousHGCalPositionsType::Float)
      types_size = sizeof(float);
    else if (cpos::types[i - 1] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
      types_size = sizeof(int32_t);
    else if (cpos::types[i - 1] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
      types_size = sizeof(uint32_t);
    else
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "Wrong HeterogeneousHGCalPositionsMapping type";
    cumsum_sizes[i] /= types_size;
  }

  for (unsigned int j = 0; j < sz.size(); ++j) {
    //setting the pointers
    if (j != 0) {
      const unsigned int jm1 = j - 1;
      const size_t shift = cumsum_sizes[j] - cumsum_sizes[jm1];
      if (cpos::types[jm1] == cpos::HeterogeneousHGCalPositionsType::Float and
          cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Float)
        select_pointer_f_(&this->posmap_, j) = select_pointer_f_(&this->posmap_, jm1) + shift;
      else if (cpos::types[jm1] == cpos::HeterogeneousHGCalPositionsType::Float and
               cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
        select_pointer_i_(&this->posmap_, j) =
            reinterpret_cast<int32_t*>(select_pointer_f_(&this->posmap_, jm1) + shift);
      else if (cpos::types[jm1] == cpos::HeterogeneousHGCalPositionsType::Int32_t and
               cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
        select_pointer_i_(&this->posmap_, j) = select_pointer_i_(&this->posmap_, jm1) + shift;
      else if (cpos::types[jm1] == cpos::HeterogeneousHGCalPositionsType::Int32_t and
               cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
        select_pointer_u_(&this->posmap_, j) =
            reinterpret_cast<uint32_t*>(select_pointer_i_(&this->posmap_, jm1) + shift);
      else
        throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
            << "Wrong HeterogeneousHGCalPositionsMapping type";
    }

    //copying the pointers' content
    if (j >=
        this->number_position_arrays)  //required due to the assymetry between cpos::HeterogeneousHGCalPositionsMapping and cpos::HGCalPositionsMapping
    {
      for (unsigned int i = cumsum_sizes[j]; i < cumsum_sizes[j + 1]; ++i) {
        unsigned int index = i - cumsum_sizes[j];
        if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Float) {
          select_pointer_f_(&this->posmap_, j)[index] =
              select_pointer_f_(cpuPos, j - this->number_position_arrays)[index];
        } else if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t) {
          select_pointer_i_(&this->posmap_, j)[index] =
              select_pointer_i_(cpuPos, j - this->number_position_arrays)[index];
        } else if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Uint32_t) {
          select_pointer_u_(&this->posmap_, j)[index] =
              select_pointer_u_(cpuPos, j - this->number_position_arrays)[index];
        } else
          throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
              << "Wrong HeterogeneousHGCalPositions type";
      }
    }
  }
}

void HeterogeneousHGCalHEFCellPositionsConditions::transfer_data_to_heterogeneous_vars_(
    const cpos::HGCalPositionsMapping* cpuPos) {
  this->posmap_.waferSize = cpuPos->waferSize;
  this->posmap_.sensorSeparation = cpuPos->sensorSeparation;
  this->posmap_.firstLayer = cpuPos->firstLayer;
  this->posmap_.lastLayer = cpuPos->lastLayer;
  this->posmap_.waferMin = cpuPos->waferMin;
  this->posmap_.waferMax = cpuPos->waferMax;
  this->nelems_posmap_ = cpuPos->detid.size();
}

std::vector<size_t> HeterogeneousHGCalHEFCellPositionsConditions::calculate_memory_bytes_(
    cpos::HGCalPositionsMapping* cpuPos) {
  size_t npointers = cpos::types.size();
  std::vector<size_t> sizes(npointers);
  for (unsigned int i = 0; i < npointers; ++i) {
    const unsigned detid_index = 4;
    const unsigned nlayers_index = 3;
    if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Float and (i == 0 or i == 1))
      sizes[i] = select_pointer_u_(cpuPos, detid_index)
                     .size();  //x and y position array will have the same size as the detid array
    else if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Float and i == 2)
      sizes[i] = select_pointer_i_(cpuPos, nlayers_index).size();  //z position's size is equal to the #layers
    else if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Float and i > 2)
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "Wrong HeterogeneousHGCalPositions type (Float)";
    else if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
      sizes[i] = select_pointer_i_(cpuPos, i - this->number_position_arrays).size();
    else if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
      sizes[i] = select_pointer_u_(cpuPos, detid_index).size();
  }

  std::vector<size_t> sizes_units(npointers);
  for (unsigned int i = 0; i < npointers; ++i) {
    if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Float)
      sizes_units[i] = sizeof(float);
    else if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
      sizes_units[i] = sizeof(int32_t);
    else if (cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
      sizes_units[i] = sizeof(uint32_t);
  }

  //element by element multiplication
  this->sizes_.resize(npointers);
  std::transform(sizes.begin(), sizes.end(), sizes_units.begin(), this->sizes_.begin(), std::multiplies<size_t>());
  return this->sizes_;
}

HeterogeneousHGCalHEFCellPositionsConditions::~HeterogeneousHGCalHEFCellPositionsConditions() {
  cudaCheck(cudaFreeHost(this->posmap_.x));
}

//I could use template specializations
//try to use std::variant in the future to avoid similar functions with different return values
float*& HeterogeneousHGCalHEFCellPositionsConditions::select_pointer_f_(
    cpos::HeterogeneousHGCalPositionsMapping* cpuObject, const unsigned int& item) const {
  switch (item) {
    case 0:
      return cpuObject->x;
    case 1:
      return cpuObject->y;
    case 2:
      return cpuObject->zLayer;
    default:
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "select_pointer_f(heterogeneous): no item (typed " << item << ").";
      return cpuObject->x;
  }
}

std::vector<float>& HeterogeneousHGCalHEFCellPositionsConditions::select_pointer_f_(
    cpos::HGCalPositionsMapping* cpuObject, const unsigned int& item) {
  switch (item) {
    case 0:
      return cpuObject->zLayer;
    default:
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "select_pointer_f(non-heterogeneous): no item (typed " << item << ").";
      return cpuObject->zLayer;
  }
}

int32_t*& HeterogeneousHGCalHEFCellPositionsConditions::select_pointer_i_(
    cpos::HeterogeneousHGCalPositionsMapping* cpuObject, const unsigned int& item) const {
  switch (item) {
    case 3:
      return cpuObject->nCellsLayer;
    case 4:
      return cpuObject->nCellsWaferUChunk;
    case 5:
      return cpuObject->nCellsHexagon;
    default:
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "select_pointer_i(heterogeneous): no item (typed " << item << ").";
      return cpuObject->nCellsHexagon;
  }
}

std::vector<int32_t>& HeterogeneousHGCalHEFCellPositionsConditions::select_pointer_i_(
    cpos::HGCalPositionsMapping* cpuObject, const unsigned int& item) {
  switch (item) {
    case 1:
      return cpuObject->nCellsLayer;
    case 2:
      return cpuObject->nCellsWaferUChunk;
    case 3:
      return cpuObject->nCellsHexagon;
    default:
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "select_pointer_i(non-heterogeneous): no item (typed " << item << ").";
      return cpuObject->nCellsHexagon;
  }
}

uint32_t*& HeterogeneousHGCalHEFCellPositionsConditions::select_pointer_u_(
    cpos::HeterogeneousHGCalPositionsMapping* cpuObject, const unsigned int& item) const {
  switch (item) {
    case 6:
      return cpuObject->detid;
    default:
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "select_pointer_u(heterogeneous): no item (typed " << item << ").";
      return cpuObject->detid;
  }
}

std::vector<uint32_t>& HeterogeneousHGCalHEFCellPositionsConditions::select_pointer_u_(
    cpos::HGCalPositionsMapping* cpuObject, const unsigned int& item) {
  switch (item) {
    case 4:
      return cpuObject->detid;
    default:
      throw cms::Exception("HeterogeneousHGCalHEFCellPositionsConditions")
          << "select_pointer_u(non-heterogeneous): no item (typed " << item << ").";
      return cpuObject->detid;
  }
}

hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct const*
HeterogeneousHGCalHEFCellPositionsConditions::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
  // cms::cuda::ESProduct<T> essentially holds an array of GPUData objects,
  // one per device. If the data have already been transferred to the
  // current device (or the transfer has been queued), the helper just
  // returns a reference to that GPUData object. Otherwise, i.e. data are
  // not yet on the current device, the helper calls the lambda to do the
  // necessary memory allocations and to queue the transfers.
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate the payload object on pinned host memory.
    cudaCheck(cudaMallocHost(&data.host, sizeof(hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct)));
    // Allocate the payload array(s) on device memory.
    cudaCheck(cudaMalloc(&(data.host->posmap.x), this->chunk_));
    // Complete the host-side information on the payload
    data.host->posmap.waferSize = this->posmap_.waferSize;
    data.host->posmap.sensorSeparation = this->posmap_.sensorSeparation;
    data.host->posmap.firstLayer = this->posmap_.firstLayer;
    data.host->posmap.lastLayer = this->posmap_.lastLayer;
    data.host->posmap.waferMax = this->posmap_.waferMax;
    data.host->posmap.waferMin = this->posmap_.waferMin;
    data.host->nelems_posmap = this->nelems_posmap_;

    //(set the pointers of the positions' mapping)
    size_t sfloat = sizeof(float);
    size_t sint32 = sizeof(int32_t);
    for (unsigned int j = 0; j < this->sizes_.size() - 1; ++j) {
      if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Float and
          cpos::types[j + 1] == cpos::HeterogeneousHGCalPositionsType::Float)
        select_pointer_f_(&(data.host->posmap), j + 1) =
            select_pointer_f_(&(data.host->posmap), j) + (this->sizes_[j] / sfloat);
      else if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Float and
               cpos::types[j + 1] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
        select_pointer_i_(&(data.host->posmap), j + 1) =
            reinterpret_cast<int32_t*>(select_pointer_f_(&(data.host->posmap), j) + (this->sizes_[j] / sfloat));
      else if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t and
               cpos::types[j + 1] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
        select_pointer_i_(&(data.host->posmap), j + 1) =
            select_pointer_i_(&(data.host->posmap), j) + (this->sizes_[j] / sint32);
      else if (cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t and
               cpos::types[j + 1] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
        select_pointer_u_(&(data.host->posmap), j + 1) =
            reinterpret_cast<uint32_t*>(select_pointer_i_(&(data.host->posmap), j) + (this->sizes_[j] / sint32));
    }

    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.device, sizeof(hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct)));

    // Transfer the payload, first the array(s) ...
    //Important: The transfer does *not* start at posmap.x because the positions are not known in the CPU side!
    size_t non_position_memory_size_to_transfer =
        this->chunk_ - this->number_position_arrays * this->nelems_posmap_ *
                           sfloat;  //size in bytes occupied by the non-position information
    cudaCheck(cudaMemcpyAsync(data.host->posmap.zLayer,
                              this->posmap_.zLayer,
                              non_position_memory_size_to_transfer,
                              cudaMemcpyHostToDevice,
                              stream));

    // ... and then the payload object
    cudaCheck(cudaMemcpyAsync(data.device,
                              data.host,
                              sizeof(hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct),
                              cudaMemcpyHostToDevice,
                              stream));

    //Fill x and y positions in the GPU
    KernelManagerHGCalCellPositions km(this->nelems_posmap_);
    km.fill_positions(data.device);
  });  //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousHGCalHEFCellPositionsConditions::GPUData::~GPUData() {
  if (host != nullptr) {
    cudaCheck(cudaFree(host->posmap.x));
    cudaCheck(cudaFreeHost(host));
  }
  cudaCheck(cudaFree(device));
}

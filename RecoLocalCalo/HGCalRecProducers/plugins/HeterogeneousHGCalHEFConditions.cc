#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFConditions.h"

HeterogeneousHGCalHEFConditionsWrapper::HeterogeneousHGCalHEFConditionsWrapper(const HGCalParameters* cpuHGCalParameters,
									       cpos::HGCalPositionsMapping* cpuPos)
{
  //HGCalParameters as defined in CMSSW
  this->sizes_params_ = calculate_memory_bytes_params_(cpuHGCalParameters);
  this->chunk_params_ = allocate_memory_params_(this->sizes_params_);
  transfer_data_to_heterogeneous_pointers_params_(this->sizes_params_, cpuHGCalParameters);
 
  //HGCalPositions as defined in hgcal_conditions::positions
  this->sizes_pos_ = calculate_memory_bytes_pos_(cpuPos);
  this->chunk_pos_ = allocate_memory_pos_(this->sizes_pos_);
  transfer_data_to_heterogeneous_pointers_pos_(this->sizes_pos_, cpuPos);
  transfer_data_to_heterogeneous_vars_pos_(cpuPos);
}

size_t HeterogeneousHGCalHEFConditionsWrapper::allocate_memory_params_(const std::vector<size_t>& sz)
{
  size_t chunk_ = std::accumulate(sz.begin(), sz.end(), 0); //total memory required in bytes
  gpuErrchk(cudaMallocHost(&this->params_.cellFineX_, chunk_));
  return chunk_;
}

size_t HeterogeneousHGCalHEFConditionsWrapper::allocate_memory_pos_(const std::vector<size_t>& sz)
{
  size_t chunk_ = std::accumulate(sz.begin(), sz.end(), 0); //total memory required in bytes
  gpuErrchk(cudaMallocHost(&this->posmap_.numberCellsHexagon, chunk_));
  return chunk_;
}

void HeterogeneousHGCalHEFConditionsWrapper::transfer_data_to_heterogeneous_pointers_params_(const std::vector<size_t>& sz, const HGCalParameters* cpuParams)
{
  //store cumulative sum in bytes and convert it to sizes in units of C++ typesHEF, i.e., number if items to be transferred to GPU
  std::vector<size_t> cumsum_sizes( sz.size()+1, 0 ); //starting with zero
  std::partial_sum(sz.begin(), sz.end(), cumsum_sizes.begin()+1);
  for(unsigned int i=1; i<cumsum_sizes.size(); ++i) //start at second element (the first is zero)
    {
      size_t typesHEFsize = 0;
      if( cpar::typesHEF[i-1] == cpar::HeterogeneousHGCalHEFParametersType::Double )
	typesHEFsize = sizeof(double);
      else if( cpar::typesHEF[i-1] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t )
	typesHEFsize = sizeof(int32_t);
      else
	edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "Wrong HeterogeneousHGCalParameters type";
      cumsum_sizes[i] /= typesHEFsize;
    }

  for(unsigned int j=0; j<sz.size(); ++j) { 

    //setting the pointers
    if(j != 0)
      {
	const unsigned int jm1 = j-1;
	const size_t shift = cumsum_sizes[j] - cumsum_sizes[jm1];
	if( cpar::typesHEF[jm1] == cpar::HeterogeneousHGCalHEFParametersType::Double and 
	    cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double )
	  select_pointer_d_(&this->params_, j) = select_pointer_d_(&this->params_, jm1) + shift;
	else if( cpar::typesHEF[jm1] == cpar::HeterogeneousHGCalHEFParametersType::Double and 
		 cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t )
	  select_pointer_i_(&this->params_, j) = reinterpret_cast<int32_t*>( select_pointer_d_(&this->params_, jm1) + shift );
      }

    //copying the pointers' content
    for(unsigned int i=cumsum_sizes[j]; i<cumsum_sizes[j+1]; ++i) 
      {
	unsigned int index = i - cumsum_sizes[j];
	if( cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double ) {
	  select_pointer_d_(&this->params_, j)[index] = select_pointer_d_(cpuParams, j)[index];
	}	  
	else if( cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t )
	  {
	    select_pointer_i_(&this->params_, j)[index] = select_pointer_i_(cpuParams, j)[index];
	  }
	else
	  edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "Wrong HeterogeneousHGCalParameters type";
      }
  }
}

void HeterogeneousHGCalHEFConditionsWrapper::transfer_data_to_heterogeneous_pointers_pos_(const std::vector<size_t>& sz, cpos::HGCalPositionsMapping* cpuPos)
{
  //store cumulative sum in bytes and convert it to sizes in units of C++ typesHEF, i.e., number if items to be transferred to GPU
  std::vector<size_t> cumsum_sizes( sz.size()+1, 0 ); //starting with zero
  std::partial_sum(sz.begin(), sz.end(), cumsum_sizes.begin()+1);
  for(unsigned int i=1; i<cumsum_sizes.size(); ++i) //start at second element (the first is zero)
    {
      size_t types_size = 0;
      if( cpos::types[i-1] == cpos::HeterogeneousHGCalPositionsType::Int32_t )
	types_size = sizeof(int32_t);
      else if( cpos::types[i-1] == cpos::HeterogeneousHGCalPositionsType::Uint32_t )
	types_size = sizeof(uint32_t);
      else
	edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "Wrong HeterogeneousHGCalPositionsMapping type";
      cumsum_sizes[i] /= types_size;
    }

  for(unsigned int j=0; j<sz.size(); ++j) { 

    //setting the pointers
    if(j != 0)
      {
	const unsigned int jm1 = j-1;
	const size_t shift = cumsum_sizes[j] - cumsum_sizes[jm1];
	if( cpos::types[jm1] == cpos::HeterogeneousHGCalPositionsType::Int32_t and 
	    cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Uint32_t )
	  select_pointer_u_(&this->posmap_, j) = reinterpret_cast<uint32_t*>( select_pointer_i_(&this->posmap_, jm1) + shift );
      }

    //copying the pointers' content
    for(unsigned int i=cumsum_sizes[j]; i<cumsum_sizes[j+1]; ++i) 
      {
	unsigned int index = i - cumsum_sizes[j];
	if( cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t ) {
	  select_pointer_i_(&this->posmap_, j)[index] = select_pointer_i_(cpuPos, j)[index];
	}	  
	else if( cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Uint32_t )
	  {
	    select_pointer_u_(&this->posmap_, j)[index] = select_pointer_u_(cpuPos, j)[index];
	  }
	else
	  edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "Wrong HeterogeneousHGCalPositions type";
      }
  }
}

void HeterogeneousHGCalHEFConditionsWrapper::transfer_data_to_heterogeneous_vars_pos_(const cpos::HGCalPositionsMapping* cpuPos) {
  this->posmap_.firstLayer = cpuPos->firstLayer;
  this->posmap_.lastLayer  = cpuPos->lastLayer;
  this->posmap_.waferMin   = cpuPos->waferMin;
  this->posmap_.waferMax   = cpuPos->waferMax;
}

std::vector<size_t> HeterogeneousHGCalHEFConditionsWrapper::calculate_memory_bytes_params_(const HGCalParameters* cpuParams) {
  size_t npointers = hgcal_conditions::parameters::typesHEF.size();
  std::vector<size_t> sizes(npointers);
  for(unsigned int i=0; i<npointers; ++i)
    {
      if(cpar::typesHEF[i] == cpar::HeterogeneousHGCalHEFParametersType::Double)
	sizes[i] = select_pointer_d_(cpuParams, i).size();
      else
	sizes[i] = select_pointer_i_(cpuParams, i).size();
    }

  std::vector<size_t> sizes_units(npointers);
  for(unsigned int i=0; i<npointers; ++i)
    {
      if(cpar::typesHEF[i] == cpar::HeterogeneousHGCalHEFParametersType::Double)
	sizes_units[i] = sizeof(double);
      else if(cpar::typesHEF[i] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t)
	sizes_units[i] = sizeof(int32_t);
    }

  //element by element multiplication
  this->sizes_params_.resize(npointers);
  std::transform( sizes.begin(), sizes.end(), sizes_units.begin(), this->sizes_params_.begin(), std::multiplies<size_t>() );
  return this->sizes_params_;
}

std::vector<size_t> HeterogeneousHGCalHEFConditionsWrapper::calculate_memory_bytes_pos_(cpos::HGCalPositionsMapping* cpuPos) {
  size_t npointers = cpos::types.size();
  std::vector<size_t> sizes(npointers);
  for(unsigned int i=0; i<npointers; ++i)
    {
      if(cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
	sizes[i] = select_pointer_i_(cpuPos, i).size();
      else if(cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
	sizes[i] = select_pointer_u_(cpuPos, i).size();
    }

  std::vector<size_t> sizes_units(npointers);
  for(unsigned int i=0; i<npointers; ++i)
    {
      if(cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Int32_t)
	sizes_units[i] = sizeof(int32_t);
      else if(cpos::types[i] == cpos::HeterogeneousHGCalPositionsType::Uint32_t)
	sizes_units[i] = sizeof(uint32_t);
    }
  
  //element by element multiplication
  this->sizes_pos_.resize(npointers);
  std::transform( sizes.begin(), sizes.end(), sizes_units.begin(), this->sizes_pos_.begin(), std::multiplies<size_t>() );
  return this->sizes_pos_;
}

HeterogeneousHGCalHEFConditionsWrapper::~HeterogeneousHGCalHEFConditionsWrapper() {
  gpuErrchk(cudaFreeHost(this->params_.cellFineX_));
}

//I could use template specializations
//try to use std::variant in the future to avoid similar functions with different return values
double*& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_d_(cpar::HeterogeneousHGCalHEFParameters* cpuObject, 
								   const unsigned int& item) const {
  switch(item) 
    {
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
  switch(item) 
    {
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
  switch(item) 
    {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_i(heterogeneous): no item.";
      return cpuObject->waferTypeL_;
    }
}

std::vector<int32_t> HeterogeneousHGCalHEFConditionsWrapper::select_pointer_i_(const HGCalParameters* cpuObject, 
									       const unsigned int& item) const {
  switch(item) 
    {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_i(non-heterogeneous): no item.";
      return cpuObject->waferTypeL_;
    }
}

int32_t*& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_i_(cpos::HeterogeneousHGCalPositionsMapping* cpuObject, 
								    const unsigned int& item) const {
  switch(item) 
    {
    case 0:
      return cpuObject->numberCellsHexagon;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_i(heterogeneous): no item.";
      return cpuObject->numberCellsHexagon;
    }
}

std::vector<int32_t>& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_i_(cpos::HGCalPositionsMapping* cpuObject, 
										const unsigned int& item) {
  switch(item) 
    {
    case 0:
      return cpuObject->numberCellsHexagon;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_i(non-heterogeneous): no item.";
      return cpuObject->numberCellsHexagon;
    }
}

uint32_t*& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_u_(cpos::HeterogeneousHGCalPositionsMapping* cpuObject, 
								    const unsigned int& item) const {
  switch(item) 
    {
    case 1:
      return cpuObject->detid;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_u(heterogeneous): no item.";
      return cpuObject->detid;
    }
}

std::vector<uint32_t>& HeterogeneousHGCalHEFConditionsWrapper::select_pointer_u_(cpos::HGCalPositionsMapping* cpuObject, 
										 const unsigned int& item) {
  switch(item) 
    {
    case 1:
      return cpuObject->detid;
    default:
      edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "select_pointer_u(non-heterogeneous): no item.";
      return cpuObject->detid;
    }
}

hgcal_conditions::HeterogeneousHEFConditionsESProduct const *HeterogeneousHGCalHEFConditionsWrapper::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
  // cms::cuda::ESProduct<T> essentially holds an array of GPUData objects,
  // one per device. If the data have already been transferred to the
  // current device (or the transfer has been queued), the helper just
  // returns a reference to that GPUData object. Otherwise, i.e. data are
  // not yet on the current device, the helper calls the lambda to do the
  // necessary memory allocations and to queue the transfers.
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream,
	  [this](GPUData& data, cudaStream_t stream)
	  {    
	    // Allocate the payload object on pinned host memory.
	    gpuErrchk(cudaMallocHost(&data.host, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct)));
	    // Allocate the payload array(s) on device memory.
	    gpuErrchk(cudaMalloc(&(data.host->params.cellFineX_), chunk_params_));
	    gpuErrchk(cudaMalloc(&(data.host->posmap.numberCellsHexagon), chunk_pos_));
	    // Complete the host-side information on the payload
	    data.host->posmap.firstLayer = this->posmap_.firstLayer;
	    data.host->posmap.lastLayer  = this->posmap_.lastLayer;
	    data.host->posmap.waferMax   = this->posmap_.waferMax;
	    data.host->posmap.waferMin   = this->posmap_.waferMin;
	    
	    // Allocate the payload object on the device memory.
	    gpuErrchk(cudaMalloc(&data.device, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct)));
	    
	    // Transfer the payload, first the array(s) ...
	    gpuErrchk(cudaMemcpyAsync(data.host->params.cellFineX_, this->params_.cellFineX_, chunk_params_, cudaMemcpyHostToDevice, stream));
	    gpuErrchk(cudaMemcpyAsync(data.host->posmap.numberCellsHexagon, this->posmap_.numberCellsHexagon, chunk_pos_, cudaMemcpyHostToDevice, stream));
	    
	    //(set the pointers of the parameters)
	    size_t sdouble = sizeof(double);
	    for(unsigned int j=0; j<this->sizes_params_.size()-1; ++j)
	      {
		if( cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double and 
		    cpar::typesHEF[j+1] == cpar::HeterogeneousHGCalHEFParametersType::Double )
		  select_pointer_d_(&(data.host->params), j+1) = select_pointer_d_(&(data.host->params), j) + (this->sizes_params_[j]/sdouble);
		else if( cpar::typesHEF[j] == cpar::HeterogeneousHGCalHEFParametersType::Double and 
			 cpar::typesHEF[j+1] == cpar::HeterogeneousHGCalHEFParametersType::Int32_t )
		  select_pointer_i_(&(data.host->params), j+1) = reinterpret_cast<int32_t*>( select_pointer_d_(&(data.host->params), j) + (this->sizes_params_[j]/sdouble) );
		else
		  edm::LogError("HeterogeneousHGCalHEFConditionsWrapper") << "compare this functions' logic with hgcal_conditions::parameters::typesHEF";
	      }

	    //(set the pointers of the positions)
	    size_t sint32 = sizeof(int32_t);
	    for(unsigned int j=0; j<this->sizes_pos_.size()-1; ++j)
	      {
		if( cpos::types[j] == cpos::HeterogeneousHGCalPositionsType::Int32_t and 
		    cpos::types[j+1] == cpos::HeterogeneousHGCalPositionsType::Uint32_t )
		  select_pointer_u_(&(data.host->posmap), j+1) = reinterpret_cast<uint32_t*>( select_pointer_i_(&(data.host->posmap), j) + (this->sizes_pos_[j]/sint32) );
	      }

	    // ... and then the payload object
	    gpuErrchk(cudaMemcpyAsync(data.device, data.host, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct), cudaMemcpyHostToDevice, stream));

	  }); //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousHGCalHEFConditionsWrapper::GPUData::~GPUData() {
  if(host != nullptr) 
    {
      gpuErrchk(cudaFree(host->params.cellFineX_));
      gpuErrchk(cudaFree(host->posmap.numberCellsHexagon));
      gpuErrchk(cudaFreeHost(host));
    }
  gpuErrchk(cudaFree(device));
}

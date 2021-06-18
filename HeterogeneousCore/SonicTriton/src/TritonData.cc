#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonMemResource.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "model_config.pb.h"

#include <sstream>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

namespace nvidia {
  namespace inferenceserver {
    //in libgrpcclient.so, but corresponding header src/core/model_config.h not available
    size_t GetDataTypeByteSize(const inference::DataType dtype);
    inference::DataType ProtocolStringToDataType(const std::string& dtype);
  }  // namespace inferenceserver
}  // namespace nvidia

//dims: kept constant, represents config.pbtxt parameters of model (converted from google::protobuf::RepeatedField to vector)
//fullShape: if batching is enabled, first entry is batch size; values can be modified
//shape: view into fullShape, excluding batch size entry
template <typename IO>
TritonData<IO>::TritonData(const std::string& name,
                           const TritonData<IO>::TensorMetadata& model_info,
                           TritonClient* client,
                           const std::string& pid)
    : name_(name),
      client_(client),
      useShm_(client_->useSharedMemory()),
      //ensure unique name for shared memory region
      shmName_(useShm_ ? pid + "_" + xput() + std::to_string(uid()) : ""),
      dims_(model_info.shape().begin(), model_info.shape().end()),
      noBatch_(client_->noBatch()),
      batchSize_(0),
      fullShape_(dims_),
      shape_(fullShape_.begin() + (noBatch_ ? 0 : 1), fullShape_.end()),
      variableDims_(anyNeg(shape_)),
      productDims_(variableDims_ ? -1 : dimProduct(shape_)),
      dname_(model_info.datatype()),
      dtype_(ni::ProtocolStringToDataType(dname_)),
      byteSize_(ni::GetDataTypeByteSize(dtype_)),
      totalByteSize_(0) {
  //create input or output object
  IO* iotmp;
  createObject(&iotmp);
  data_.reset(iotmp);
}

template <>
void TritonInputData::createObject(nic::InferInput** ioptr) {
  nic::InferInput::Create(ioptr, name_, fullShape_, dname_);
}

template <>
void TritonOutputData::createObject(nic::InferRequestedOutput** ioptr) {
  nic::InferRequestedOutput::Create(ioptr, name_);
  //another specialization for output: can't use shared memory if output size is not known
  useShm_ &= !variableDims_;
}

template <>
std::string TritonInputData::xput() const {
  return "input";
}

template <>
std::string TritonOutputData::xput() const {
  return "output";
}

template <typename IO>
nic::InferenceServerGrpcClient* TritonData<IO>::client() {
  return client_->client();
}

//setters
template <typename IO>
void TritonData<IO>::setShape(const TritonData<IO>::ShapeType& newShape) {
  for (unsigned i = 0; i < newShape.size(); ++i) {
    setShape(i, newShape[i]);
  }
}

template <typename IO>
void TritonData<IO>::setShape(unsigned loc, int64_t val) {
  unsigned locFull = fullLoc(loc);

  //check boundary
  if (locFull >= fullShape_.size())
    throw cms::Exception("TritonDataError")
        << name_ << " setShape(): dimension " << locFull << " out of bounds (" << fullShape_.size() << ")";

  if (val != fullShape_[locFull]) {
    if (dims_[locFull] == -1)
      fullShape_[locFull] = val;
    else
      throw cms::Exception("TritonDataError")
          << name_ << " setShape(): attempt to change value of non-variable shape dimension " << loc;
  }
}

template <typename IO>
void TritonData<IO>::setBatchSize(unsigned bsize) {
  batchSize_ = bsize;
  if (!noBatch_)
    fullShape_[0] = batchSize_;
}

template <typename IO>
void TritonData<IO>::computeSizes() {
  sizeShape_ = sizeShape();
  byteSizePerBatch_ = byteSize_ * sizeShape_;
  totalByteSize_ = byteSizePerBatch_ * batchSize_;
}
template <typename IO>
void TritonData<IO>::resetSizes() {
  sizeShape_ = 0;
  byteSizePerBatch_ = 0;
  totalByteSize_ = 0;
}

//create a memory resource if none exists;
//otherwise, reuse the memory resource, resizing it if necessary
template <typename IO>
void TritonData<IO>::updateMem(size_t size) {
  if (!memResource_ or size > memResource_->size()) {
    if (useShm_ and client_->serverType() == TritonServerType::LocalCPU) {
      //avoid unnecessarily throwing in destructor
      if (memResource_)
        memResource_->close();
      //need to destroy before constructing new instance because shared memory key will be reused
      memResource_.reset();
      memResource_ = std::make_shared<TritonCpuShmResource<IO>>(this, shmName_, size);
    }
#ifdef TRITON_ENABLE_GPU
    else if (useShm_ and client_->serverType() == TritonServerType::LocalGPU) {
      //avoid unnecessarily throwing in destructor
      if (memResource_)
        memResource_->close();
      //need to destroy before constructing new instance because shared memory key will be reused
      memResource_.reset();
      memResource_ = std::make_shared<TritonGpuShmResource<IO>>(this, shmName_, size);
    }
#endif
    //for remote/heap, size increases don't matter
    else if (!memResource_)
      memResource_ = std::make_shared<TritonHeapResource<IO>>(this, shmName_, size);
  }
}

//io accessors
template <>
template <typename DT>
TritonInputContainer<DT> TritonInputData::allocate(bool reserve) {
  //automatically creates a vector for each batch entry (if batch size known)
  auto ptr = std::make_shared<TritonInput<DT>>(batchSize_);
  if (reserve and !anyNeg(shape_)) {
    computeSizes();
    for (auto& vec : *ptr) {
      vec.reserve(sizeShape_);
    }
  }
  return ptr;
}

template <>
template <typename DT>
void TritonInputData::toServer(TritonInputContainer<DT> ptr) {
  const auto& data_in = *ptr;

  //check batch size
  if (data_in.size() != batchSize_) {
    throw cms::Exception("TritonDataError") << name_ << " toServer(): input vector has size " << data_in.size()
                                            << " but specified batch size is " << batchSize_;
  }

  //shape must be specified for variable dims or if batch size changes
  data_->SetShape(fullShape_);

  if (byteSize_ != sizeof(DT))
    throw cms::Exception("TritonDataError") << name_ << " toServer(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";

  computeSizes();
  updateMem(totalByteSize_);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    memResource_->copyInput(data_in[i0].data(), i0 * byteSizePerBatch_);
  }
  memResource_->set();

  //keep input data in scope
  holder_ = ptr;
}

//sets up shared memory for outputs, if possible
template <>
void TritonOutputData::prepare() {
  computeSizes();
  updateMem(totalByteSize_);
  memResource_->set();
}

template <>
template <typename DT>
TritonOutput<DT> TritonOutputData::fromServer() const {
  if (!result_) {
    throw cms::Exception("TritonDataError") << name_ << " fromServer(): missing result";
  }

  if (byteSize_ != sizeof(DT)) {
    throw cms::Exception("TritonDataError") << name_ << " fromServer(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";
  }

  const uint8_t* r0 = memResource_->copyOutput();
  const DT* r1 = reinterpret_cast<const DT*>(r0);

  TritonOutput<DT> dataOut;
  dataOut.reserve(batchSize_);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    auto offset = i0 * sizeShape_;
    dataOut.emplace_back(r1 + offset, r1 + offset + sizeShape_);
  }

  return dataOut;
}

template <>
void TritonInputData::reset() {
  holder_.reset();
  data_->Reset();
  //reset shape
  if (variableDims_) {
    for (unsigned i = 0; i < shape_.size(); ++i) {
      unsigned locFull = fullLoc(i);
      fullShape_[locFull] = dims_[locFull];
    }
  }
  resetSizes();
}

template <>
void TritonOutputData::reset() {
  result_.reset();
  holder_.reset();
  resetSizes();
}

//explicit template instantiation declarations
template class TritonData<nic::InferInput>;
template class TritonData<nic::InferRequestedOutput>;

template TritonInputContainer<float> TritonInputData::allocate(bool reserve);
template TritonInputContainer<int64_t> TritonInputData::allocate(bool reserve);

template void TritonInputData::toServer(TritonInputContainer<float> data_in);
template void TritonInputData::toServer(TritonInputContainer<int64_t> data_in);

template TritonOutput<float> TritonOutputData::fromServer() const;

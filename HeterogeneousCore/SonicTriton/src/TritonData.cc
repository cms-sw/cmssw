#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "model_config.pb.h"

#include <cstring>
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
TritonData<IO>::TritonData(const std::string& name, const TritonData<IO>::TensorMetadata& model_info, bool noBatch)
    : name_(name),
      dims_(model_info.shape().begin(), model_info.shape().end()),
      noBatch_(noBatch),
      batchSize_(0),
      fullShape_(dims_),
      shape_(fullShape_.begin() + (noBatch_ ? 0 : 1), fullShape_.end()),
      variableDims_(anyNeg(shape_)),
      productDims_(variableDims_ ? -1 : dimProduct(shape_)),
      dname_(model_info.datatype()),
      dtype_(ni::ProtocolStringToDataType(dname_)),
      byteSize_(ni::GetDataTypeByteSize(dtype_)) {
  //create input or output object
  IO* iotmp;
  createObject(&iotmp);
  data_.reset(iotmp);
}

template <>
void TritonInputData::createObject(nic::InferInput** ioptr) const {
  nic::InferInput::Create(ioptr, name_, fullShape_, dname_);
}

template <>
void TritonOutputData::createObject(nic::InferRequestedOutput** ioptr) const {
  nic::InferRequestedOutput::Create(ioptr, name_);
}

//setters
template <typename IO>
bool TritonData<IO>::setShape(const TritonData<IO>::ShapeType& newShape, bool canThrow) {
  bool result = true;
  for (unsigned i = 0; i < newShape.size(); ++i) {
    result &= setShape(i, newShape[i], canThrow);
  }
  return result;
}

template <typename IO>
bool TritonData<IO>::setShape(unsigned loc, int64_t val, bool canThrow) {
  std::stringstream msg;
  unsigned full_loc = loc + (noBatch_ ? 0 : 1);

  //check boundary
  if (full_loc >= fullShape_.size()) {
    msg << name_ << " setShape(): dimension " << full_loc << " out of bounds (" << fullShape_.size() << ")";
    if (canThrow)
      throw cms::Exception("TritonDataError") << msg.str();
    else {
      edm::LogWarning("TritonDataWarning") << msg.str();
      return false;
    }
  }

  if (val != fullShape_[full_loc]) {
    if (dims_[full_loc] == -1) {
      fullShape_[full_loc] = val;
      return true;
    } else {
      msg << name_ << " setShape(): attempt to change value of non-variable shape dimension " << loc;
      if (canThrow)
        throw cms::Exception("TritonDataError") << msg.str();
      else {
        edm::LogWarning("TritonDataError") << msg.str();
        return false;
      }
    }
  }

  return true;
}

template <typename IO>
void TritonData<IO>::setBatchSize(unsigned bsize) {
  batchSize_ = bsize;
  if (!noBatch_)
    fullShape_[0] = batchSize_;
}

//io accessors
template <>
template <typename DT>
void TritonInputData::toServer(std::shared_ptr<TritonInput<DT>> ptr) {
  const auto& data_in = *ptr;

  //check batch size
  if (data_in.size() != batchSize_) {
    throw cms::Exception("TritonDataError") << name_ << " input(): input vector has size " << data_in.size()
                                            << " but specified batch size is " << batchSize_;
  }

  //shape must be specified for variable dims or if batch size changes
  data_->SetShape(fullShape_);

  if (byteSize_ != sizeof(DT))
    throw cms::Exception("TritonDataError") << name_ << " input(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";

  int64_t nInput = sizeShape();
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    const DT* arr = data_in[i0].data();
    triton_utils::throwIfError(data_->AppendRaw(reinterpret_cast<const uint8_t*>(arr), nInput * byteSize_),
                               name_ + " input(): unable to set data for batch entry " + std::to_string(i0));
  }

  //keep input data in scope
  holder_ = std::move(ptr);
}

template <>
template <typename DT>
TritonOutput<DT> TritonOutputData::fromServer() const {
  if (!result_) {
    throw cms::Exception("TritonDataError") << name_ << " output(): missing result";
  }

  if (byteSize_ != sizeof(DT)) {
    throw cms::Exception("TritonDataError") << name_ << " output(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";
  }

  uint64_t nOutput = sizeShape();
  TritonOutput<DT> dataOut;
  const uint8_t* r0;
  size_t contentByteSize;
  size_t expectedContentByteSize = nOutput * byteSize_ * batchSize_;
  triton_utils::throwIfError(result_->RawData(name_, &r0, &contentByteSize), "output(): unable to get raw");
  if (contentByteSize != expectedContentByteSize) {
    throw cms::Exception("TritonDataError") << name_ << " output(): unexpected content byte size " << contentByteSize
                                            << " (expected " << expectedContentByteSize << ")";
  }

  const DT* r1 = reinterpret_cast<const DT*>(r0);
  dataOut.reserve(batchSize_);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    auto offset = i0 * nOutput;
    dataOut.emplace_back(r1 + offset, r1 + offset + nOutput);
  }

  return dataOut;
}

template <>
void TritonInputData::reset() {
  data_->Reset();
  holder_.reset();
}

template <>
void TritonOutputData::reset() {
  result_.reset();
}

//explicit template instantiation declarations
template class TritonData<nic::InferInput>;
template class TritonData<nic::InferRequestedOutput>;

template void TritonInputData::toServer(std::shared_ptr<TritonInput<float>> data_in);
template void TritonInputData::toServer(std::shared_ptr<TritonInput<int64_t>> data_in);

template TritonOutput<float> TritonOutputData::fromServer() const;

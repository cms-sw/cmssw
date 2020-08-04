#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "model_config.pb.h"

#include <cstring>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

namespace nvidia {
  namespace inferenceserver {
    //in librequest.so, but corresponding header src/core/model_config.h not available
    size_t GetDataTypeByteSize(const DataType dtype);
  }  // namespace inferenceserver
}  // namespace nvidia

template <typename IO>
TritonData<IO>::TritonData(const std::string& name, std::shared_ptr<IO> data)
    : name_(name), data_(std::move(data)), batchSize_(0) {
  //convert google::protobuf::RepeatedField to vector
  const auto& dimsTmp = data_->Dims();
  dims_.assign(dimsTmp.begin(), dimsTmp.end());

  //check if variable dimensions
  variableDims_ = anyNeg(dims_);
  if (variableDims_)
    productDims_ = -1;
  else
    productDims_ = dimProduct(dims_);

  dtype_ = data_->DType();
  dname_ = ni::DataType_Name(dtype_);
  //get byte size for input conversion
  byteSize_ = ni::GetDataTypeByteSize(dtype_);
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

  //shape must be specified for variable dims
  if (variableDims_) {
    if (shape_.size() != dims_.size()) {
      throw cms::Exception("TritonDataError")
          << name_ << " input(): incorrect or missing shape (" << triton_utils::printColl(shape_)
          << ") for model with variable dimensions (" << triton_utils::printColl(dims_) << ")";
    } else {
      triton_utils::throwIfError(data_->SetShape(shape_), name_ + " input(): unable to set input shape");
    }
  }

  if (byteSize_ != sizeof(DT))
    throw cms::Exception("TritonDataError") << name_ << " input(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";

  int64_t nInput = sizeShape();
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    const DT* arr = data_in[i0].data();
    triton_utils::throwIfError(data_->SetRaw(reinterpret_cast<const uint8_t*>(arr), nInput * byteSize_),
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

  //shape must be specified for variable dims
  if (variableDims_) {
    if (shape_.size() != dims_.size()) {
      throw cms::Exception("TritonDataError")
          << name_ << " output(): incorrect or missing shape (" << triton_utils::printColl(shape_)
          << ") for model with variable dimensions (" << triton_utils::printColl(dims_) << ")";
    }
  }

  if (byteSize_ != sizeof(DT)) {
    throw cms::Exception("TritonDataError") << name_ << " output(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";
  }

  uint64_t nOutput = sizeShape();
  TritonOutput<DT> dataOut;
  dataOut.reserve(batchSize_);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    const uint8_t* r0;
    size_t contentByteSize;
    triton_utils::throwIfError(result_->GetRaw(i0, &r0, &contentByteSize),
                               "output(): unable to get raw for entry " + std::to_string(i0));
    if (contentByteSize != nOutput * byteSize_) {
      throw cms::Exception("TritonDataError") << name_ << " output(): unexpected content byte size " << contentByteSize
                                              << " (expected " << nOutput * byteSize_ << ")";
    }
    const DT* r1 = reinterpret_cast<const DT*>(r0);
    dataOut.emplace_back(r1, r1 + nOutput);
  }

  return dataOut;
}

template <>
void TritonInputData::reset() {
  shape_.clear();
  data_->Reset();
  holder_.reset();
}

template <>
void TritonOutputData::reset() {
  shape_.clear();
  result_.reset();
}

//explicit template instantiation declarations
template class TritonData<nic::InferContext::Input>;
template class TritonData<nic::InferContext::Output>;

template void TritonInputData::toServer(std::shared_ptr<TritonInput<float>> data_in);
template void TritonInputData::toServer(std::shared_ptr<TritonInput<int64_t>> data_in);

template TritonOutput<float> TritonOutputData::fromServer() const;

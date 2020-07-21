#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonUtils.h"
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
    : name_(name), data_(data), batchSize_(0) {
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
void TritonInputData::toServer(std::shared_ptr<std::vector<DT>> ptr) {
  const auto& data_in = *(ptr.get());

  //shape must be specified for variable dims
  if (variableDims_) {
    if (shape_.size() != dims_.size()) {
      throw cms::Exception("TritonDataError")
          << name_ << " input(): incorrect or missing shape (" << TritonUtils::printVec(shape_)
          << ") for model with variable dimensions (" << TritonUtils::printVec(dims_) << ")";
    } else {
      TritonUtils::wrap(data_->SetShape(shape_), name_ + " input(): unable to set input shape");
    }
  }

  if (byteSize_ != sizeof(DT))
    throw cms::Exception("TritonDataError") << name_ << " input(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";

  int64_t nInput = sizeShape();
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    const DT* arr = &(data_in[i0 * nInput]);
    TritonUtils::wrap(data_->SetRaw(reinterpret_cast<const uint8_t*>(arr), nInput * byteSize_),
                      name_ + " input(): unable to set data for batch entry " + std::to_string(i0));
  }

  //keep input data in scope
  callback_ = [ptr]() { return; };
}

template <>
template <typename DT>
void TritonOutputData::fromServer(std::vector<DT>& dataOut) const {
  if (!result_) {
    throw cms::Exception("TritonDataError") << name_ << " output(): missing result";
  }

  //shape must be specified for variable dims
  if (variableDims_) {
    if (shape_.size() != dims_.size()) {
      throw cms::Exception("TritonDataError")
          << name_ << " output(): incorrect or missing shape (" << TritonUtils::printVec(shape_)
          << ") for model with variable dimensions (" << TritonUtils::printVec(dims_) << ")";
    }
  }

  if (byteSize_ != sizeof(DT)) {
    throw cms::Exception("TritonDataError") << name_ << " output(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";
  }

  int64_t nOutput = sizeShape();
  dataOut.resize(nOutput * batchSize_, 0);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    const uint8_t* r0;
    size_t contentByteSize;
    TritonUtils::wrap(result_->GetRaw(i0, &r0, &contentByteSize),
                      "output(): unable to get raw for entry " + std::to_string(i0));
    std::memcpy(&(dataOut[i0 * nOutput]), r0, nOutput * byteSize_);
  }
}

template <>
void TritonInputData::reset() {
  shape_.clear();
  data_->Reset();
  if (callback_)
    callback_();
}

template <>
void TritonOutputData::reset() {
  shape_.clear();
  result_.reset();
}

//explicit template instantiation declarations
template class TritonData<nic::InferContext::Input>;
template class TritonData<nic::InferContext::Output>;

template void TritonInputData::toServer(std::shared_ptr<std::vector<float>> data_in);
template void TritonInputData::toServer(std::shared_ptr<std::vector<int64_t>> data_in);

template void TritonOutputData::fromServer(std::vector<float>& data_out) const;

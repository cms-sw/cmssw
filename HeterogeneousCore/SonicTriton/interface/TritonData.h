#ifndef HeterogeneousCore_SonicTriton_TritonData
#define HeterogeneousCore_SonicTriton_TritonData

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <algorithm>
#include <memory>

#include "request_grpc.h"

//store all the info needed for triton input and output
template <typename IO>
class TritonData {
public:
  using Result = nvidia::inferenceserver::client::InferContext::Result;

  //constructor
  TritonData(const std::string& name, std::shared_ptr<IO> data);

  //some members can be modified
  std::vector<int64_t>& shape() { return shape_; }
  void reset();
  void setBatchSize(unsigned bsize) { batchSize_ = bsize; }
  void setResult(std::unique_ptr<Result> result) { result_ = std::move(result); }

  //io accessors
  template <typename DT>
  void toServer(std::shared_ptr<std::vector<DT>> ptr);
  template <typename DT>
  void fromServer(std::vector<DT>& data_out) const;

  //const accessors
  const std::shared_ptr<IO>& data() const { return data_; }
  const std::vector<int64_t>& dims() const { return dims_; }
  const std::vector<int64_t>& shape() const { return shape_.empty() ? dims() : shape_; }
  int64_t byteSize() const { return byteSize_; }
  const std::string& dname() const { return dname_; }
  unsigned batchSize() const { return batchSize_; }

  //utilities
  bool variableDims() const { return variableDims_; }
  int64_t sizeDims() const { return productDims_; }
  //default to dims if shape isn't filled
  int64_t sizeShape() const { return shape_.empty() ? sizeDims() : dimProduct(shape_); }

private:
  //helpers
  bool anyNeg(const std::vector<int64_t>& vec) const {
    return std::any_of(vec.begin(), vec.end(), [](int64_t i) { return i < 0; });
  }
  int64_t dimProduct(const std::vector<int64_t>& vec) const {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  }

  //members
  std::string name_;
  std::shared_ptr<IO> data_;
  std::vector<int64_t> dims_;
  bool variableDims_;
  int64_t productDims_;
  nvidia::inferenceserver::DataType dtype_;
  std::string dname_;
  int64_t byteSize_;
  std::vector<int64_t> shape_;
  unsigned batchSize_;
  std::function<void(void)> callback_;
  std::unique_ptr<Result> result_;
};

using TritonInputData = TritonData<nvidia::inferenceserver::client::InferContext::Input>;
using TritonInputMap = std::unordered_map<std::string, TritonInputData>;
using TritonOutputData = TritonData<nvidia::inferenceserver::client::InferContext::Output>;
using TritonOutputMap = std::unordered_map<std::string, TritonOutputData>;

//avoid "explicit specialization after instantiation" error
template <>
template <typename DT>
void TritonInputData::toServer(std::shared_ptr<std::vector<DT>> ptr);
template <>
template <typename DT>
void TritonOutputData::fromServer(std::vector<DT>& dataOut) const;
template <>
void TritonInputData::reset();
template <>
void TritonOutputData::reset();

//explicit template instantiation declarations
extern template class TritonData<nvidia::inferenceserver::client::InferContext::Input>;
extern template class TritonData<nvidia::inferenceserver::client::InferContext::Output>;

#endif

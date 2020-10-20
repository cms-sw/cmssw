#ifndef HeterogeneousCore_SonicTriton_TritonData
#define HeterogeneousCore_SonicTriton_TritonData

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Span.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <memory>
#include <any>

#include "grpc_client.h"
#include "grpc_service.pb.h"

//forward declaration
class TritonClient;

//aliases for local input and output types
template <typename DT>
using TritonInput = std::vector<std::vector<DT>>;
template <typename DT>
using TritonOutput = std::vector<edm::Span<const DT*>>;

//store all the info needed for triton input and output
template <typename IO>
class TritonData {
public:
  using Result = nvidia::inferenceserver::client::InferResult;
  using TensorMetadata = inference::ModelMetadataResponse_TensorMetadata;
  using ShapeType = std::vector<int64_t>;
  using ShapeView = edm::Span<ShapeType::const_iterator>;

  //constructor
  TritonData(const std::string& name, const TensorMetadata& model_info, bool noBatch);

  //some members can be modified
  bool setShape(const ShapeType& newShape) { return setShape(newShape, true); }
  bool setShape(unsigned loc, int64_t val) { return setShape(loc, val, true); }

  //io accessors
  template <typename DT>
  void toServer(std::shared_ptr<TritonInput<DT>> ptr);
  template <typename DT>
  TritonOutput<DT> fromServer() const;

  //const accessors
  const ShapeView& shape() const { return shape_; }
  int64_t byteSize() const { return byteSize_; }
  const std::string& dname() const { return dname_; }
  unsigned batchSize() const { return batchSize_; }

  //utilities
  bool variableDims() const { return variableDims_; }
  int64_t sizeDims() const { return productDims_; }
  //default to dims if shape isn't filled
  int64_t sizeShape() const { return variableDims_ ? dimProduct(shape_) : sizeDims(); }

private:
  friend class TritonClient;

  //private accessors only used by client
  bool setShape(const ShapeType& newShape, bool canThrow);
  bool setShape(unsigned loc, int64_t val, bool canThrow);
  void setBatchSize(unsigned bsize);
  void reset();
  void setResult(std::shared_ptr<Result> result) { result_ = result; }
  IO* data() { return data_.get(); }

  //helpers
  bool anyNeg(const ShapeView& vec) const {
    return std::any_of(vec.begin(), vec.end(), [](int64_t i) { return i < 0; });
  }
  int64_t dimProduct(const ShapeView& vec) const {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  }
  void createObject(IO** ioptr) const;

  //members
  std::string name_;
  std::shared_ptr<IO> data_;
  const ShapeType dims_;
  bool noBatch_;
  unsigned batchSize_;
  ShapeType fullShape_;
  ShapeView shape_;
  bool variableDims_;
  int64_t productDims_;
  std::string dname_;
  inference::DataType dtype_;
  int64_t byteSize_;
  std::any holder_;
  std::shared_ptr<Result> result_;
};

using TritonInputData = TritonData<nvidia::inferenceserver::client::InferInput>;
using TritonInputMap = std::unordered_map<std::string, TritonInputData>;
using TritonOutputData = TritonData<nvidia::inferenceserver::client::InferRequestedOutput>;
using TritonOutputMap = std::unordered_map<std::string, TritonOutputData>;

//avoid "explicit specialization after instantiation" error
template <>
template <typename DT>
void TritonInputData::toServer(std::shared_ptr<TritonInput<DT>> ptr);
template <>
template <typename DT>
TritonOutput<DT> TritonOutputData::fromServer() const;
template <>
void TritonInputData::reset();
template <>
void TritonOutputData::reset();
template <>
void TritonInputData::createObject(nvidia::inferenceserver::client::InferInput** ioptr) const;
template <>
void TritonOutputData::createObject(nvidia::inferenceserver::client::InferRequestedOutput** ioptr) const;

//explicit template instantiation declarations
extern template class TritonData<nvidia::inferenceserver::client::InferInput>;
extern template class TritonData<nvidia::inferenceserver::client::InferRequestedOutput>;

#endif

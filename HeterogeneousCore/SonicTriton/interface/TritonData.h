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
#include <atomic>

#include "grpc_client.h"
#include "grpc_service.pb.h"

//forward declaration
class TritonClient;
template <typename IO>
class TritonMemResource;
template <typename IO>
class TritonHeapResource;
template <typename IO>
class TritonCpuShmResource;
#ifdef TRITON_ENABLE_GPU
template <typename IO>
class TritonGpuShmResource;
#endif

//aliases for local input and output types
template <typename DT>
using TritonInput = std::vector<std::vector<DT>>;
template <typename DT>
using TritonOutput = std::vector<edm::Span<const DT*>>;

//other useful typdefs
template <typename DT>
using TritonInputContainer = std::shared_ptr<TritonInput<DT>>;

//store all the info needed for triton input and output
template <typename IO>
class TritonData {
public:
  using Result = nvidia::inferenceserver::client::InferResult;
  using TensorMetadata = inference::ModelMetadataResponse_TensorMetadata;
  using ShapeType = std::vector<int64_t>;
  using ShapeView = edm::Span<ShapeType::const_iterator>;

  //constructor
  TritonData(const std::string& name, const TensorMetadata& model_info, TritonClient* client, const std::string& pid);

  //some members can be modified
  void setShape(const ShapeType& newShape);
  void setShape(unsigned loc, int64_t val);

  //io accessors
  template <typename DT>
  TritonInputContainer<DT> allocate(bool reserve = true);
  template <typename DT>
  void toServer(TritonInputContainer<DT> ptr);
  void prepare();
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
  friend class TritonMemResource<IO>;
  friend class TritonHeapResource<IO>;
  friend class TritonCpuShmResource<IO>;
#ifdef TRITON_ENABLE_GPU
  friend class TritonGpuShmResource<IO>;
#endif

  //private accessors only used internally or by client
  unsigned fullLoc(unsigned loc) const { return loc + (noBatch_ ? 0 : 1); }
  void setBatchSize(unsigned bsize);
  void reset();
  void setResult(std::shared_ptr<Result> result) { result_ = result; }
  IO* data() { return data_.get(); }
  void updateMem(size_t size);
  void computeSizes();
  void resetSizes();
  nvidia::inferenceserver::client::InferenceServerGrpcClient* client();

  //helpers
  bool anyNeg(const ShapeView& vec) const {
    return std::any_of(vec.begin(), vec.end(), [](int64_t i) { return i < 0; });
  }
  int64_t dimProduct(const ShapeView& vec) const {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  }
  void createObject(IO** ioptr);
  //generates a unique id number for each instance of the class
  unsigned uid() const {
    static std::atomic<unsigned> uid{0};
    return ++uid;
  }
  std::string xput() const;

  //members
  std::string name_;
  std::shared_ptr<IO> data_;
  TritonClient* client_;
  bool useShm_;
  std::string shmName_;
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
  size_t sizeShape_;
  size_t byteSizePerBatch_;
  size_t totalByteSize_;
  std::shared_ptr<void> holder_;
  std::shared_ptr<TritonMemResource<IO>> memResource_;
  std::shared_ptr<Result> result_;
};

using TritonInputData = TritonData<nvidia::inferenceserver::client::InferInput>;
using TritonInputMap = std::unordered_map<std::string, TritonInputData>;
using TritonOutputData = TritonData<nvidia::inferenceserver::client::InferRequestedOutput>;
using TritonOutputMap = std::unordered_map<std::string, TritonOutputData>;

//avoid "explicit specialization after instantiation" error
template <>
std::string TritonInputData::xput() const;
template <>
std::string TritonOutputData::xput() const;
template <>
template <typename DT>
TritonInputContainer<DT> TritonInputData::allocate(bool reserve);
template <>
template <typename DT>
void TritonInputData::toServer(std::shared_ptr<TritonInput<DT>> ptr);
template <>
void TritonOutputData::prepare();
template <>
template <typename DT>
TritonOutput<DT> TritonOutputData::fromServer() const;
template <>
void TritonInputData::reset();
template <>
void TritonOutputData::reset();
template <>
void TritonInputData::createObject(nvidia::inferenceserver::client::InferInput** ioptr);
template <>
void TritonOutputData::createObject(nvidia::inferenceserver::client::InferRequestedOutput** ioptr);

//explicit template instantiation declarations
extern template class TritonData<nvidia::inferenceserver::client::InferInput>;
extern template class TritonData<nvidia::inferenceserver::client::InferRequestedOutput>;

#endif

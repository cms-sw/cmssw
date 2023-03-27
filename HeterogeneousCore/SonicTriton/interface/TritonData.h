#ifndef HeterogeneousCore_SonicTriton_TritonData
#define HeterogeneousCore_SonicTriton_TritonData

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Span.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <memory>
#include <atomic>
#include <typeinfo>

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
//NOTE: this class is not const-thread-safe, and should only be used with stream or one modules
//(generally recommended for SONIC, but especially necessary here)
template <typename IO>
class TritonData {
public:
  using Result = triton::client::InferResult;
  using TensorMetadata = inference::ModelMetadataResponse_TensorMetadata;
  using ShapeType = std::vector<int64_t>;
  using ShapeView = edm::Span<ShapeType::const_iterator>;

  //constructor
  TritonData(const std::string& name, const TensorMetadata& model_info, TritonClient* client, const std::string& pid);

  //some members can be modified
  void setShape(const ShapeType& newShape, unsigned entry = 0);
  void setShape(unsigned loc, int64_t val, unsigned entry = 0);

  //io accessors
  template <typename DT>
  TritonInputContainer<DT> allocate(bool reserve = true);
  template <typename DT>
  void toServer(TritonInputContainer<DT> ptr);
  void prepare();
  template <typename DT>
  TritonOutput<DT> fromServer() const;

  //const accessors
  const ShapeView& shape(unsigned entry = 0) const { return entries_.at(entry).shape_; }
  int64_t byteSize() const { return byteSize_; }
  const std::string& dname() const { return dname_; }

  //utilities
  bool variableDims() const { return variableDims_; }
  int64_t sizeDims() const { return productDims_; }
  //default to dims if shape isn't filled
  int64_t sizeShape(unsigned entry = 0) const {
    return variableDims_ ? dimProduct(entries_.at(entry).shape_) : sizeDims();
  }

private:
  friend class TritonClient;
  friend class TritonMemResource<IO>;
  friend class TritonHeapResource<IO>;
  friend class TritonCpuShmResource<IO>;
#ifdef TRITON_ENABLE_GPU
  friend class TritonGpuShmResource<IO>;
#endif

  //group together all relevant information for a single request
  //helpful for organizing multi-request ragged batching case
  class TritonDataEntry {
  public:
    //constructors
    TritonDataEntry(const ShapeType& dims, bool noOuterDim, const std::string& name, const std::string& dname)
        : fullShape_(dims),
          shape_(fullShape_.begin() + (noOuterDim ? 0 : 1), fullShape_.end()),
          sizeShape_(0),
          byteSizePerBatch_(0),
          totalByteSize_(0),
          offset_(0),
          output_(nullptr) {
      //create input or output object
      IO* iotmp;
      createObject(&iotmp, name, dname);
      data_.reset(iotmp);
    }
    //default needed to be able to use std::vector resize()
    TritonDataEntry()
        : shape_(fullShape_.begin(), fullShape_.end()),
          sizeShape_(0),
          byteSizePerBatch_(0),
          totalByteSize_(0),
          offset_(0),
          output_(nullptr) {}

  private:
    friend class TritonData<IO>;
    friend class TritonClient;
    friend class TritonMemResource<IO>;
    friend class TritonHeapResource<IO>;
    friend class TritonCpuShmResource<IO>;
#ifdef TRITON_ENABLE_GPU
    friend class TritonGpuShmResource<IO>;
#endif

    //accessors
    void createObject(IO** ioptr, const std::string& name, const std::string& dname);
    void computeSizes(int64_t shapeSize, int64_t byteSize, int64_t batchSize);

    //members
    ShapeType fullShape_;
    ShapeView shape_;
    size_t sizeShape_, byteSizePerBatch_, totalByteSize_;
    std::shared_ptr<IO> data_;
    std::shared_ptr<Result> result_;
    unsigned offset_;
    const uint8_t* output_;
  };

  //private accessors only used internally or by client
  void checkShm() {}
  unsigned fullLoc(unsigned loc) const;
  void reset();
  void setResult(std::shared_ptr<Result> result, unsigned entry = 0) { entries_[entry].result_ = result; }
  IO* data(unsigned entry = 0) { return entries_[entry].data_.get(); }
  void updateMem(size_t size);
  void computeSizes();
  triton::client::InferenceServerGrpcClient* client();
  template <typename DT>
  void checkType() const {
    if (!triton_utils::checkType<DT>(dtype_))
      throw cms::Exception("TritonDataError")
          << name_ << ": inconsistent data type " << typeid(DT).name() << " for " << dname_;
  }

  //helpers
  bool anyNeg(const ShapeView& vec) const {
    return std::any_of(vec.begin(), vec.end(), [](int64_t i) { return i < 0; });
  }
  int64_t dimProduct(const ShapeView& vec) const {
    //lambda treats negative dimensions as 0 to avoid overflows
    return std::accumulate(
        vec.begin(), vec.end(), 1, [](int64_t dim1, int64_t dim2) { return dim1 * std::max(0l, dim2); });
  }
  //generates a unique id number for each instance of the class
  unsigned uid() const {
    static std::atomic<unsigned> uid{0};
    return ++uid;
  }
  std::string xput() const;
  void addEntry(unsigned entry);
  void addEntryImpl(unsigned entry);

  //members
  std::string name_;
  TritonClient* client_;
  bool useShm_;
  std::string shmName_;
  const ShapeType dims_;
  bool variableDims_;
  int64_t productDims_;
  std::string dname_;
  inference::DataType dtype_;
  int64_t byteSize_;
  std::vector<TritonDataEntry> entries_;
  size_t totalByteSize_;
  //can be modified in otherwise-const fromServer() method in TritonMemResource::copyOutput():
  //TritonMemResource holds a non-const pointer to an instance of this class
  //so that TritonOutputGpuShmResource can store data here
  std::shared_ptr<void> holder_;
  std::shared_ptr<TritonMemResource<IO>> memResource_;
  //can be modified in otherwise-const fromServer() method to prevent multiple calls
  CMS_SA_ALLOW mutable bool done_{};
};

using TritonInputData = TritonData<triton::client::InferInput>;
using TritonInputMap = std::unordered_map<std::string, TritonInputData>;
using TritonOutputData = TritonData<triton::client::InferRequestedOutput>;
using TritonOutputMap = std::unordered_map<std::string, TritonOutputData>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputData::TritonDataEntry::createObject(triton::client::InferInput** ioptr,
                                                    const std::string& name,
                                                    const std::string& dname);
template <>
void TritonOutputData::TritonDataEntry::createObject(triton::client::InferRequestedOutput** ioptr,
                                                     const std::string& name,
                                                     const std::string& dname);
template <>
void TritonOutputData::checkShm();
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

//explicit template instantiation declarations
extern template class TritonData<triton::client::InferInput>;
extern template class TritonData<triton::client::InferRequestedOutput>;

#endif

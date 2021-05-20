#ifndef HeterogeneousCore_SonicTriton_TritonMemResource
#define HeterogeneousCore_SonicTriton_TritonMemResource

#include <string>

//forward declaration
template <typename IO>
class TritonData;
struct cudaIpcMemHandle_st;
typedef cudaIpcMemHandle_st cudaIpcMemHandle_t;

//base class for memory operations
template <typename IO>
class TritonMemResource {
public:
  TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  virtual ~TritonMemResource() {}
  uint8_t* addr() { return addr_; }
  size_t size() const { return size_; }
  bool status() const { return status_; }
  virtual void copy(const void* values, size_t offset) {}
  virtual void copy(const uint8_t** values) {}
  virtual bool set(bool canThrow);

protected:
  //member variables
  TritonData<IO>* data_;
  std::string name_;
  size_t size_;
  uint8_t* addr_;
  bool status_;
};

template <typename IO>
class TritonHeapResource : public TritonMemResource<IO> {
public:
  TritonHeapResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  ~TritonHeapResource() override {}
  void copy(const void* values, size_t offset) override {}
  void copy(const uint8_t** values) override {}
  bool set(bool canThrow) override { return true; }
};

template <typename IO>
class TritonCpuShmResource : public TritonMemResource<IO> {
public:
  TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  ~TritonCpuShmResource() override;
  void copy(const void* values, size_t offset) override {}
  void copy(const uint8_t** values) override {}
};

template <typename IO>
class TritonGpuShmResource : public TritonMemResource<IO> {
public:
  TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  ~TritonGpuShmResource() override;
  void copy(const void* values, size_t offset) override {}
  void copy(const uint8_t** values) override {}

protected:
  int deviceId_;
  std::shared_ptr<cudaIpcMemHandle_t> handle_;
};

using TritonInputHeapResource = TritonHeapResource<nvidia::inferenceserver::client::InferInput>;
using TritonInputCpuShmResource = TritonCpuShmResource<nvidia::inferenceserver::client::InferInput>;
using TritonInputGpuShmResource = TritonGpuShmResource<nvidia::inferenceserver::client::InferInput>;
using TritonOutputHeapResource = TritonHeapResource<nvidia::inferenceserver::client::InferRequestedOutput>;
using TritonOutputCpuShmResource = TritonCpuShmResource<nvidia::inferenceserver::client::InferRequestedOutput>;
using TritonOutputGpuShmResource = TritonGpuShmResource<nvidia::inferenceserver::client::InferRequestedOutput>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputHeapResource::copy(const void* values, size_t offset);
template <>
void TritonInputCpuShmResource::copy(const void* values, size_t offset);
template <>
void TritonInputGpuShmResource::copy(const void* values, size_t offset);
template <>
void TritonOutputHeapResource::copy(const uint8_t** values);
template <>
void TritonOutputCpuShmResource::copy(const uint8_t** values);
template <>
void TritonOutputGpuShmResource::copy(const uint8_t** values);

#endif

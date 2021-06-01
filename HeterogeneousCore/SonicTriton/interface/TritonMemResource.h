#ifndef HeterogeneousCore_SonicTriton_TritonMemResource
#define HeterogeneousCore_SonicTriton_TritonMemResource

#include <string>
#include <memory>

#include "grpc_client.h"

//forward declaration
template <typename IO>
class TritonData;

//base class for memory operations
template <typename IO>
class TritonMemResource {
public:
  TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size);
  virtual ~TritonMemResource() {}
  uint8_t* addr() { return addr_; }
  size_t size() const { return size_; }
  virtual void close() {}
  //used for input
  virtual void copyInput(const void* values, size_t offset) {}
  //used for output
  virtual const uint8_t* copyOutput() { return nullptr; }
  virtual void set();

protected:
  //member variables
  TritonData<IO>* data_;
  std::string name_;
  size_t size_;
  uint8_t* addr_;
  bool closed_;
};

template <typename IO>
class TritonHeapResource : public TritonMemResource<IO> {
public:
  TritonHeapResource(TritonData<IO>* data, const std::string& name, size_t size);
  ~TritonHeapResource() override {}
  void copyInput(const void* values, size_t offset) override {}
  const uint8_t* copyOutput() override { return nullptr; }
  void set() override {}
};

template <typename IO>
class TritonCpuShmResource : public TritonMemResource<IO> {
public:
  TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size);
  ~TritonCpuShmResource() override;
  void close() override;
  void copyInput(const void* values, size_t offset) override {}
  const uint8_t* copyOutput() override { return nullptr; }
};

using TritonInputHeapResource = TritonHeapResource<nvidia::inferenceserver::client::InferInput>;
using TritonInputCpuShmResource = TritonCpuShmResource<nvidia::inferenceserver::client::InferInput>;
using TritonOutputHeapResource = TritonHeapResource<nvidia::inferenceserver::client::InferRequestedOutput>;
using TritonOutputCpuShmResource = TritonCpuShmResource<nvidia::inferenceserver::client::InferRequestedOutput>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputHeapResource::copyInput(const void* values, size_t offset);
template <>
void TritonInputCpuShmResource::copyInput(const void* values, size_t offset);
template <>
const uint8_t* TritonOutputHeapResource::copyOutput();
template <>
const uint8_t* TritonOutputCpuShmResource::copyOutput();

#ifdef TRITON_ENABLE_GPU
#include "cuda_runtime_api.h"

template <typename IO>
class TritonGpuShmResource : public TritonMemResource<IO> {
public:
  TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size);
  ~TritonGpuShmResource() override;
  void close() override;
  void copyInput(const void* values, size_t offset) override {}
  const uint8_t* copyOutput() override { return nullptr; }

protected:
  int deviceId_;
  std::shared_ptr<cudaIpcMemHandle_t> handle_;
};

using TritonInputGpuShmResource = TritonGpuShmResource<nvidia::inferenceserver::client::InferInput>;
using TritonOutputGpuShmResource = TritonGpuShmResource<nvidia::inferenceserver::client::InferRequestedOutput>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputGpuShmResource::copyInput(const void* values, size_t offset);
template <>
const uint8_t* TritonOutputGpuShmResource::copyOutput();
#endif

#endif

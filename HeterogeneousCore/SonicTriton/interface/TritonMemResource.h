#ifndef HeterogeneousCore_SonicTriton_TritonMemResource
#define HeterogeneousCore_SonicTriton_TritonMemResource

#include <string>
#include <memory>

#include "grpc_client.h"

#include "cuda_runtime_api.h"

//forward declaration
template <typename IO>
class TritonData;

//base class for memory operations
template <typename IO>
class TritonMemResource {
public:
  TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  virtual ~TritonMemResource() {}
  uint8_t* addr() { return addr_; }
  size_t size() const { return size_; }
  bool status() const { return status_; }
  //used for input
  virtual void copyInput(const void* values, size_t offset) {}
  //used for output
  virtual const uint8_t* copyOutput() { return nullptr; }
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
  void copyInput(const void* values, size_t offset) override {}
  const uint8_t* copyOutput() override { return nullptr; }
  bool set(bool canThrow) override { return true; }
};

template <typename IO>
class TritonCpuShmResource : public TritonMemResource<IO> {
public:
  TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  ~TritonCpuShmResource() override;
  void copyInput(const void* values, size_t offset) override {}
  const uint8_t* copyOutput() override { return nullptr; }
};

template <typename IO>
class TritonGpuShmResource : public TritonMemResource<IO> {
public:
  TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow);
  ~TritonGpuShmResource() override;
  void copyInput(const void* values, size_t offset) override {}
  const uint8_t* copyOutput() override { return nullptr; }

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
void TritonInputHeapResource::copyInput(const void* values, size_t offset);
template <>
void TritonInputCpuShmResource::copyInput(const void* values, size_t offset);
template <>
void TritonInputGpuShmResource::copyInput(const void* values, size_t offset);
template <>
const uint8_t* TritonOutputHeapResource::copyOutput();
template <>
const uint8_t* TritonOutputCpuShmResource::copyOutput();
template <>
const uint8_t* TritonOutputGpuShmResource::copyOutput();

#endif

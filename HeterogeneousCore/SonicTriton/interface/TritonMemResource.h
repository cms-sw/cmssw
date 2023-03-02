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
  virtual void copyInput(const void* values, size_t offset, unsigned entry) {}
  //used for output
  virtual void copyOutput() {}
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
  void copyInput(const void* values, size_t offset, unsigned entry) override {}
  void copyOutput() override {}
  void set() override {}
};

template <typename IO>
class TritonCpuShmResource : public TritonMemResource<IO> {
public:
  TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size);
  ~TritonCpuShmResource() override;
  void close() override;
  void copyInput(const void* values, size_t offset, unsigned entry) override {}
  void copyOutput() override {}

protected:
  size_t sizeOrig_;
};

using TritonInputHeapResource = TritonHeapResource<triton::client::InferInput>;
using TritonInputCpuShmResource = TritonCpuShmResource<triton::client::InferInput>;
using TritonOutputHeapResource = TritonHeapResource<triton::client::InferRequestedOutput>;
using TritonOutputCpuShmResource = TritonCpuShmResource<triton::client::InferRequestedOutput>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputHeapResource::copyInput(const void* values, size_t offset, unsigned entry);
template <>
void TritonInputCpuShmResource::copyInput(const void* values, size_t offset, unsigned entry);
template <>
void TritonOutputHeapResource::copyOutput();
template <>
void TritonOutputCpuShmResource::copyOutput();

#ifdef TRITON_ENABLE_GPU
#include "cuda_runtime_api.h"

template <typename IO>
class TritonGpuShmResource : public TritonMemResource<IO> {
public:
  TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size);
  ~TritonGpuShmResource() override;
  void close() override;
  void copyInput(const void* values, size_t offset, unsigned entry) override {}
  void copyOutput() override {}

protected:
  int deviceId_;
  std::shared_ptr<cudaIpcMemHandle_t> handle_;
};

using TritonInputGpuShmResource = TritonGpuShmResource<triton::client::InferInput>;
using TritonOutputGpuShmResource = TritonGpuShmResource<triton::client::InferRequestedOutput>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputGpuShmResource::copyInput(const void* values, size_t offset, unsigned entry);
template <>
void TritonOutputGpuShmResource::copyOutput();
#endif

#endif

#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonMemResource.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

template <typename IO>
TritonMemResource<IO>::TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size)
    : data_(data), name_(name), size_(size), addr_(nullptr), closed_(false) {}

template <typename IO>
void TritonMemResource<IO>::set() {
  triton_utils::throwIfError(data_->data_->SetSharedMemory(name_, data_->totalByteSize_, 0),
                             "unable to set shared memory (" + name_ + ")");
}

template <typename IO>
TritonHeapResource<IO>::TritonHeapResource(TritonData<IO>* data, const std::string& name, size_t size)
    : TritonMemResource<IO>(data, name, size) {}

template <>
void TritonInputHeapResource::copyInput(const void* values, size_t offset) {
  triton_utils::throwIfError(
      data_->data_->AppendRaw(reinterpret_cast<const uint8_t*>(values), data_->byteSizePerBatch_),
      data_->name_ + " toServer(): unable to set data for batch entry " +
          std::to_string(offset / data_->byteSizePerBatch_));
}

template <>
const uint8_t* TritonOutputHeapResource::copyOutput() {
  size_t contentByteSize;
  const uint8_t* values;
  triton_utils::throwIfError(data_->result_->RawData(data_->name_, &values, &contentByteSize),
                             data_->name_ + " fromServer(): unable to get raw");
  if (contentByteSize != data_->totalByteSize_) {
    throw cms::Exception("TritonDataError") << data_->name_ << " fromServer(): unexpected content byte size "
                                            << contentByteSize << " (expected " << data_->totalByteSize_ << ")";
  }
  return values;
}

//shared memory helpers based on:
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc (cpu)
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/simple_grpc_cudashm_client.cc (gpu)

template <typename IO>
TritonCpuShmResource<IO>::TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size)
    : TritonMemResource<IO>(data, name, size) {
  //get shared memory region descriptor
  int shm_fd = shm_open(this->name_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (shm_fd == -1)
    throw cms::Exception("TritonError") << "unable to get shared memory descriptor for key: " + this->name_;

  //extend shared memory object
  int res = ftruncate(shm_fd, this->size_);
  if (res == -1)
    throw cms::Exception("TritonError") << "unable to initialize shared memory key " + this->name_ +
                                               " to requested size: " + std::to_string(this->size_);

  //map to process address space
  constexpr size_t offset(0);
  this->addr_ = (uint8_t*)mmap(nullptr, this->size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (this->addr_ == MAP_FAILED)
    throw cms::Exception("TritonError") << "unable to map to process address space for shared memory key: " +
                                               this->name_;

  //close descriptor
  if (::close(shm_fd) == -1)
    throw cms::Exception("TritonError") << "unable to close descriptor for shared memory key: " + this->name_;

  triton_utils::throwIfError(this->data_->client()->RegisterSystemSharedMemory(this->name_, this->name_, this->size_),
                             "unable to register shared memory region: " + this->name_);
}

template <typename IO>
TritonCpuShmResource<IO>::~TritonCpuShmResource<IO>() {
  close();
}

template <typename IO>
void TritonCpuShmResource<IO>::close() {
  if (this->closed_)
    return;

  triton_utils::throwIfError(this->data_->client()->UnregisterSystemSharedMemory(this->name_),
                             "unable to unregister shared memory region: " + this->name_);

  //unmap
  int tmp_fd = munmap(this->addr_, this->size_);
  if (tmp_fd == -1)
    throw cms::Exception("TritonError") << "unable to munmap for shared memory key: " << this->name_;

  //unlink
  int shm_fd = shm_unlink(this->name_.c_str());
  if (shm_fd == -1)
    throw cms::Exception("TritonError") << "unable to unlink for shared memory key: " << this->name_;

  this->closed_ = true;
}

template <>
void TritonInputCpuShmResource::copyInput(const void* values, size_t offset) {
  std::memcpy(addr_ + offset, values, data_->byteSizePerBatch_);
}

template <>
const uint8_t* TritonOutputCpuShmResource::copyOutput() {
  return addr_;
}

template class TritonHeapResource<nic::InferInput>;
template class TritonCpuShmResource<nic::InferInput>;
template class TritonHeapResource<nic::InferRequestedOutput>;
template class TritonCpuShmResource<nic::InferRequestedOutput>;

#ifdef TRITON_ENABLE_GPU
template <typename IO>
TritonGpuShmResource<IO>::TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size)
    : TritonMemResource<IO>(data, name, size), deviceId_(0), handle_(std::make_shared<cudaIpcMemHandle_t>()) {
  //todo: get server device id somehow?
  cudaCheck(cudaSetDevice(deviceId_), "unable to set device ID to " + std::to_string(deviceId_));
  cudaCheck(cudaMalloc((void**)&this->addr_, this->size_), "unable to allocate GPU memory for key: " + this->name_);
  cudaCheck(cudaIpcGetMemHandle(handle_.get(), this->addr_), "unable to get IPC handle for key: " + this->name_);
  triton_utils::throwIfError(
      this->data_->client()->RegisterCudaSharedMemory(this->name_, *handle_, deviceId_, this->size_),
      "unable to register CUDA shared memory region: " + this->name_);
}

template <typename IO>
TritonGpuShmResource<IO>::~TritonGpuShmResource<IO>() {
  close();
}

template <typename IO>
void TritonGpuShmResource<IO>::close() {
  if (this->closed_)
    return;
  triton_utils::throwIfError(this->data_->client()->UnregisterCudaSharedMemory(this->name_),
                             "unable to unregister CUDA shared memory region: " + this->name_);
  cudaCheck(cudaFree(this->addr_), "unable to free GPU memory for key: " + this->name_);
  this->closed_ = true;
}

template <>
void TritonInputGpuShmResource::copyInput(const void* values, size_t offset) {
  cudaCheck(
      cudaMemcpy(addr_ + offset, values, data_->byteSizePerBatch_, cudaMemcpyHostToDevice),
      data_->name_ + " toServer(): unable to memcpy " + std::to_string(data_->byteSizePerBatch_) + " bytes to GPU");
}

template <>
const uint8_t* TritonOutputGpuShmResource::copyOutput() {
  //copy back from gpu, keep in scope
  auto ptr = std::make_shared<std::vector<uint8_t>>(data_->totalByteSize_);
  cudaCheck(
      cudaMemcpy(ptr->data(), addr_, data_->totalByteSize_, cudaMemcpyDeviceToHost),
      data_->name_ + " fromServer(): unable to memcpy " + std::to_string(data_->totalByteSize_) + " bytes from GPU");
  data_->holder_ = ptr;
  return ptr->data();
}

template class TritonGpuShmResource<nic::InferInput>;
template class TritonGpuShmResource<nic::InferRequestedOutput>;
#endif

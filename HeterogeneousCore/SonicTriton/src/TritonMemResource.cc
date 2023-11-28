#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonMemResource.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace tc = triton::client;

template <typename IO>
TritonMemResource<IO>::TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size)
    : data_(data), name_(name), size_(size), addr_(nullptr), closed_(false) {}

template <typename IO>
void TritonMemResource<IO>::set() {
  for (auto& entry : data_->entries_) {
    TRITON_THROW_IF_ERROR(entry.data_->SetSharedMemory(name_, entry.totalByteSize_, entry.offset_),
                          "unable to set shared memory (" + name_ + ")");
  }
}

template <typename IO>
TritonHeapResource<IO>::TritonHeapResource(TritonData<IO>* data, const std::string& name, size_t size)
    : TritonMemResource<IO>(data, name, size) {}

template <>
void TritonInputHeapResource::copyInput(const void* values, size_t offset, unsigned entry) {
  TRITON_THROW_IF_ERROR(data_->entries_[entry].data_->AppendRaw(reinterpret_cast<const uint8_t*>(values),
                                                                data_->entries_[entry].byteSizePerBatch_),
                        data_->name_ + " toServer(): unable to set data for batch entry " +
                            (data_->entries_.size() > 1 ? std::to_string(entry)
                             : data_->entries_[entry].byteSizePerBatch_
                                 ? std::to_string(offset / data_->entries_[entry].byteSizePerBatch_)
                                 : ""));
}

template <>
void TritonOutputHeapResource::copyOutput() {
  size_t contentByteSize = 0;
  for (auto& entry : data_->entries_) {
    size_t contentByteSizeEntry(0);
    if (entry.totalByteSize_ > 0)
      TRITON_THROW_IF_ERROR(entry.result_->RawData(data_->name_, &entry.output_, &contentByteSizeEntry),
                            data_->name_ + " fromServer(): unable to get raw");
    contentByteSize += contentByteSizeEntry;
  }
  if (contentByteSize != data_->totalByteSize_) {
    throw cms::Exception("TritonDataError") << data_->name_ << " fromServer(): unexpected content byte size "
                                            << contentByteSize << " (expected " << data_->totalByteSize_ << ")";
  }
}

//shared memory helpers based on:
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc (cpu)
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/simple_grpc_cudashm_client.cc (gpu)

template <typename IO>
TritonCpuShmResource<IO>::TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size)
    : TritonMemResource<IO>(data, name, size), sizeOrig_(size) {
  //mmap of size zero is required to fail by POSIX, but still need to have some shared memory region available for Triton
  this->size_ = std::max<size_t>(this->size_, 1);

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

  TRITON_THROW_IF_ERROR(this->data_->client()->RegisterSystemSharedMemory(this->name_, this->name_, this->size_),
                        "unable to register shared memory region: " + this->name_);
}

template <typename IO>
TritonCpuShmResource<IO>::~TritonCpuShmResource() {
  close();
}

template <typename IO>
void TritonCpuShmResource<IO>::close() {
  if (this->closed_)
    return;

  TRITON_THROW_IF_ERROR(this->data_->client()->UnregisterSystemSharedMemory(this->name_),
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
void TritonInputCpuShmResource::copyInput(const void* values, size_t offset, unsigned entry) {
  if (sizeOrig_ > 0)
    std::memcpy(addr_ + offset, values, data_->entries_[entry].byteSizePerBatch_);
}

template <>
void TritonOutputCpuShmResource::copyOutput() {
  for (auto& entry : data_->entries_) {
    entry.output_ = addr_ + entry.offset_;
  }
}

template class TritonHeapResource<tc::InferInput>;
template class TritonCpuShmResource<tc::InferInput>;
template class TritonHeapResource<tc::InferRequestedOutput>;
template class TritonCpuShmResource<tc::InferRequestedOutput>;

#ifdef TRITON_ENABLE_GPU
template <typename IO>
TritonGpuShmResource<IO>::TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size)
    : TritonMemResource<IO>(data, name, size), deviceId_(0), handle_(std::make_shared<cudaIpcMemHandle_t>()) {
  //todo: get server device id somehow?
  cudaCheck(cudaSetDevice(deviceId_), "unable to set device ID to " + std::to_string(deviceId_));
  cudaCheck(cudaMalloc((void**)&this->addr_, this->size_), "unable to allocate GPU memory for key: " + this->name_);
  cudaCheck(cudaIpcGetMemHandle(handle_.get(), this->addr_), "unable to get IPC handle for key: " + this->name_);
  TRITON_THROW_IF_ERROR(this->data_->client()->RegisterCudaSharedMemory(this->name_, *handle_, deviceId_, this->size_),
                        "unable to register CUDA shared memory region: " + this->name_);
}

template <typename IO>
TritonGpuShmResource<IO>::~TritonGpuShmResource() {
  close();
}

template <typename IO>
void TritonGpuShmResource<IO>::close() {
  if (this->closed_)
    return;
  TRITON_THROW_IF_ERROR(this->data_->client()->UnregisterCudaSharedMemory(this->name_),
                        "unable to unregister CUDA shared memory region: " + this->name_);
  cudaCheck(cudaFree(this->addr_), "unable to free GPU memory for key: " + this->name_);
  this->closed_ = true;
}

template <>
void TritonInputGpuShmResource::copyInput(const void* values, size_t offset, unsigned entry) {
  cudaCheck(cudaMemcpy(addr_ + offset, values, data_->entries_[entry].byteSizePerBatch_, cudaMemcpyHostToDevice),
            data_->name_ + " toServer(): unable to memcpy " + std::to_string(data_->entries_[entry].byteSizePerBatch_) +
                " bytes to GPU");
}

template <>
void TritonOutputGpuShmResource::copyOutput() {
  //copy back from gpu, keep in scope
  auto ptr = std::make_shared<std::vector<uint8_t>>(data_->totalByteSize_);
  cudaCheck(
      cudaMemcpy(ptr->data(), addr_, data_->totalByteSize_, cudaMemcpyDeviceToHost),
      data_->name_ + " fromServer(): unable to memcpy " + std::to_string(data_->totalByteSize_) + " bytes from GPU");
  data_->holder_ = ptr;
  for (auto& entry : data_->entries_) {
    entry.output_ = ptr->data() + entry.offset_;
  }
}

template class TritonGpuShmResource<tc::InferInput>;
template class TritonGpuShmResource<tc::InferRequestedOutput>;
#endif

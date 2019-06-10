#ifndef HeterogeneousCore_CUDAServices_CUDAService_h
#define HeterogeneousCore_CUDAServices_CUDAService_h

#include <utility>
#include <vector>

#include <cuda/api_wrappers.h>

#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ConfigurationDescriptions;
}

namespace cudaserviceimpl {
  template <typename T>
  struct make_device_unique_selector { using non_array = cudautils::device::unique_ptr<T>; };
  template <typename T>
  struct make_device_unique_selector<T[]> { using unbounded_array = cudautils::device::unique_ptr<T[]>; };
  template <typename T, size_t N>
  struct make_device_unique_selector<T[N]> { struct bounded_array {}; };

  template <typename T>
  struct make_host_unique_selector { using non_array = cudautils::host::unique_ptr<T>; };
  template <typename T>
  struct make_host_unique_selector<T[]> { using unbounded_array = cudautils::host::unique_ptr<T[]>; };
  template <typename T, size_t N>
  struct make_host_unique_selector<T[N]> { struct bounded_array {}; };
}

/**
 * TODO:
 * - CUDA stream management?
 *   * Not really needed until we want to pass CUDA stream objects from one module to another
 *   * Which is not really needed until we want to go for "streaming mode"
 *   * Until that framework's inter-module synchronization is safe (but not necessarily optimal)
 * - Management of (preallocated) memory?
 */
class CUDAService {
public:
  CUDAService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);
  ~CUDAService();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  bool enabled() const { return enabled_; }

  int numberOfDevices() const { return numberOfDevices_; }

  // major, minor
  std::pair<int, int> computeCapability(int device) { return computeCapabilities_.at(device); }

  // Returns the id of device with most free memory. If none is found, returns -1.
  int deviceWithMostFreeMemory() const;

  // Set the current device
  void setCurrentDevice(int device) const;

  // Get the current device
  int getCurrentDevice() const;

  // Allocate device memory
  template <typename T>
  typename cudaserviceimpl::make_device_unique_selector<T>::non_array
  make_device_unique(cuda::stream_t<>& stream) {
    static_assert(std::is_trivially_constructible<T>::value, "Allocating with non-trivial constructor on the device memory is not supported");
    int dev = getCurrentDevice();
    void *mem = allocate_device(dev, sizeof(T), stream);
    return typename cudaserviceimpl::make_device_unique_selector<T>::non_array(reinterpret_cast<T *>(mem),
                                                                               cudautils::device::impl::DeviceDeleter([this, dev](void *ptr) {
                                                                                   this->free_device(dev, ptr);
                                                                                 }));
  }

  template <typename T>
  typename cudaserviceimpl::make_device_unique_selector<T>::unbounded_array
  make_device_unique(size_t n, cuda::stream_t<>& stream) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value, "Allocating with non-trivial constructor on the device memory is not supported");
    int dev = getCurrentDevice();
    void *mem = allocate_device(dev, n*sizeof(element_type), stream);
    return typename cudaserviceimpl::make_device_unique_selector<T>::unbounded_array(reinterpret_cast<element_type *>(mem),
                                                                                     cudautils::device::impl::DeviceDeleter([this, dev](void *ptr) {
                                                                                         this->free_device(dev, ptr);
                                                                                       }));
  }

  template <typename T, typename ...Args>
  typename cudaserviceimpl::make_device_unique_selector<T>::bounded_array
  make_device_unique(Args&&...) = delete;

  // Allocate pinned host memory
  template <typename T>
  typename cudaserviceimpl::make_host_unique_selector<T>::non_array
  make_host_unique(cuda::stream_t<>& stream) {
    static_assert(std::is_trivially_constructible<T>::value, "Allocating with non-trivial constructor on the pinned host memory is not supported");
    void *mem = allocate_host(sizeof(T), stream);
    return typename cudaserviceimpl::make_host_unique_selector<T>::non_array(reinterpret_cast<T *>(mem),
                                                                             cudautils::host::impl::HostDeleter([this](void *ptr) {
                                                                                 this->free_host(ptr);
                                                                               }));
  }

  template <typename T>
  typename cudaserviceimpl::make_host_unique_selector<T>::unbounded_array
  make_host_unique(size_t n, cuda::stream_t<>& stream) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value, "Allocating with non-trivial constructor on the pinned host memory is not supported");
    void *mem = allocate_host(n*sizeof(element_type), stream);
    return typename cudaserviceimpl::make_host_unique_selector<T>::unbounded_array(reinterpret_cast<element_type *>(mem),
                                                                                   cudautils::host::impl::HostDeleter([this](void *ptr) {
                                                                                       this->free_host(ptr);
                                                                                     }));
  }

  template <typename T, typename ...Args>
  typename cudaserviceimpl::make_host_unique_selector<T>::bounded_array
  make_host_unique(Args&&...) = delete;
  
  // Free device memory (to be called from unique_ptr)
  void free_device(int device, void *ptr);

  // Free pinned host memory (to be called from unique_ptr)
  void free_host(void *ptr);

  // Gets a (cached) CUDA stream for the current device. The stream
  // will be returned to the cache by the shared_ptr destructor.
  std::shared_ptr<cuda::stream_t<>> getCUDAStream();

  // Gets a (cached) CUDA event for the current device. The event
  // will be returned to the cache by the shared_ptr destructor.
  std::shared_ptr<cuda::event_t> getCUDAEvent();

private:
  // PIMPL to hide details of allocator
  struct Allocator;
  std::unique_ptr<Allocator> allocator_;
  void *allocate_device(int dev, size_t nbytes, cuda::stream_t<>& stream);
  void *allocate_host(size_t nbytes, cuda::stream_t<>& stream);

  // PIMPL to hide details of the CUDA stream cache
  struct CUDAStreamCache;
  std::unique_ptr<CUDAStreamCache> cudaStreamCache_;

  // PIMPL to hide details of the CUDA event cache
  struct CUDAEventCache;
  std::unique_ptr<CUDAEventCache> cudaEventCache_;

  int numberOfDevices_ = 0;
  std::vector<std::pair<int, int>> computeCapabilities_;
  bool enabled_ = false;
};

#endif

#ifndef HeterogeneousCore_AlpakaInterface_interface_MoveToDeviceCache_h
#define HeterogeneousCore_AlpakaInterface_interface_MoveToDeviceCache_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/CopyToDeviceCache.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

namespace cms::alpakatools {
  namespace detail {
    // By default copy the host object with CopyToDevice<T>
    //
    // Doing with template specialization (rather than
    // std::conditional_t and if constexpr) because the
    // CopyToDevice<THostObject>::copyAsync() is ill-defined e.g. for
    // PortableCollection on host device
    template <typename TDevice, typename THostObject>
    class MoveToDeviceCacheImpl {
    public:
      using HostObject = THostObject;
      using Impl = CopyToDeviceCacheImpl<TDevice, THostObject>;
      using DeviceObject = typename Impl::DeviceObject;

      MoveToDeviceCacheImpl(HostObject&& srcObject) : impl_(srcObject) {}

      DeviceObject const& get(size_t i) const { return impl_.get(i); }

    private:
      Impl impl_;
    };

    // For host device, move the host object instead
    template <typename THostObject>
    class MoveToDeviceCacheImpl<alpaka_common::DevHost, THostObject> {
    public:
      using HostObject = THostObject;
      using DeviceObject = HostObject;

      MoveToDeviceCacheImpl(HostObject&& srcObject) : data_(std::move(srcObject)) {}

      DeviceObject const& get(size_t i) const { return data_; }

    private:
      HostObject data_;
    };
  }  // namespace detail

  /**
   * This class template implements a cache for data that is moved
   * from the host (of type THostObject) to all the devices
   * corresponding to the TDevice device type.
   *
   * The host-side object to be moved is given as an argument to the
   * class constructor. The constructor uses the
   * CopyToDevice<THostObject> class template to copy the data to the
   * devices, and waits for the data copies to finish, i.e. the
   * constructor is synchronous wrt. the data copies. The "move" is
   * achieved by requiring the constructor argument to be an rvalue
   * reference.
   *
   * Note that the host object type is required to be non-copyable.
   * This is to avoid easy mistakes with objects that follow copy
   * semantics of std::shared_ptr (that includes Alpaka buffers), that
   * would allow the source memory buffer to be used via another copy
   * during the asynchronous data copy to the device.
   *
   * The device-side object corresponding to the THostObject (actual
   * type is the return type of CopyToDevice<THostObject>::copyAsync())
   * can be obtained with get() member function, that has either the
   * queue or device argument.
   */
  template <typename TDevice, typename THostObject>
    requires alpaka::isDevice<TDevice>
  class MoveToDeviceCache {
  public:
    using Device = TDevice;
    using HostObject = THostObject;
    using Impl = detail::MoveToDeviceCacheImpl<Device, HostObject>;
    using DeviceObject = typename Impl::DeviceObject;

    static_assert(not(std::is_copy_constructible_v<HostObject> or std::is_copy_assignable_v<HostObject>),
                  "The data object to be moved to device must not be copyable.");

    MoveToDeviceCache(HostObject&& srcData) : data_(std::move(srcData)) {}

    DeviceObject const& get(Device const& dev) const { return data_.get(alpaka::getNativeHandle(dev)); }

    template <typename TQueue>
    DeviceObject const& get(TQueue const& queue) const {
      return get(alpaka::getDev(queue));
    }

  private:
    Impl data_;
  };
}  // namespace cms::alpakatools

#endif

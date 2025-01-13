#ifndef HeterogeneousCore_AlpakaInterface_interface_CopyToDeviceCache_h
#define HeterogeneousCore_AlpakaInterface_interface_CopyToDeviceCache_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
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
    class CopyToDeviceCacheImpl {
    public:
      using Device = TDevice;
      using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;
      using HostObject = THostObject;
      using Copy = CopyToDevice<HostObject>;
      using DeviceObject = decltype(Copy::copyAsync(std::declval<Queue&>(), std::declval<HostObject const&>()));

      CopyToDeviceCacheImpl(HostObject const& srcObject) {
        using Platform = alpaka::Platform<Device>;
        auto const& devices = cms::alpakatools::devices<Platform>();
        std::vector<std::shared_ptr<Queue>> queues;
        queues.reserve(devices.size());
        data_.reserve(devices.size());
        for (auto const& dev : devices) {
          auto queue = getQueueCache<Queue>().get(dev);
          data_.emplace_back(Copy::copyAsync(*queue, srcObject));
          queues.emplace_back(std::move(queue));
        }
        for (auto& queuePtr : queues) {
          alpaka::wait(*queuePtr);
        }
      }

      DeviceObject const& get(size_t i) const { return data_[i]; }

    private:
      std::vector<DeviceObject> data_;
    };

    // For host device, copy the host object directly instead
    template <typename THostObject>
    class CopyToDeviceCacheImpl<alpaka_common::DevHost, THostObject> {
    public:
      using HostObject = THostObject;
      using DeviceObject = HostObject;

      CopyToDeviceCacheImpl(HostObject const& srcObject) : data_(srcObject) {}

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
   * The host-side object to be copied is given as an argument to the
   * class constructor. The constructor uses the
   * CopyToDevice<THostObject> class template to perfom the copy, and
   * waits for the data copies to finish, i.e. the constructor is
   * synchronous wrt. the data copies.
   *
   * The device-side object corresponding to the THostObject (actual
   * type is the return type of CopyToDevice<THostObject>::copyAsync())
   * can be obtained with get() member function, that has either the
   * queue or device argument.
   */
  template <typename TDevice, typename THostObject>
    requires alpaka::isDevice<TDevice>
  class CopyToDeviceCache {
    using Device = TDevice;
    using HostObject = THostObject;
    using Impl = detail::CopyToDeviceCacheImpl<Device, HostObject>;
    using DeviceObject = typename Impl::DeviceObject;

  public:
    CopyToDeviceCache(THostObject const& srcData) : data_(srcData) {}

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

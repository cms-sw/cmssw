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
    template <typename TDev, typename TQueue, typename THostObject>
    class CopyToDeviceCacheImpl {
    public:
      using Device = TDev;
      using Queue = TQueue;
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
    template <typename TQueue, typename THostObject>
    class CopyToDeviceCacheImpl<alpaka_common::DevHost, TQueue, THostObject> {
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
   * This class template implements a cache for data that is copied
   * from the host (of type THostObject) to all the devices
   * corresponding the TQueue queue type.
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
   *
   * TODO: In principle it would be better to template over Device,
   * but then we'd need a way to have a "default queue" type for each
   * Device in order to infer the return type of
   * CopyToDevice::copyAsync(). Alternatively, the template over
   * TQueue could be removed by moving the class definition to
   * ALPAKA_ACCELERATOR_NAMESPACE.
   */
  template <typename TQueue, typename THostObject>
  class CopyToDeviceCache {
    using Queue = TQueue;
    using Device = alpaka::Dev<Queue>;
    using HostObject = THostObject;
    using Impl = detail::CopyToDeviceCacheImpl<Device, Queue, HostObject>;
    using DeviceObject = typename Impl::DeviceObject;

  public:
    CopyToDeviceCache(THostObject const& srcData) : data_(srcData) {}

    // TODO: I could make this function to return the contained object
    // in case of alpaka buffer, PortableObject, or PortableCollection
    // (in PortableCollection case it would be the View)
    DeviceObject const& get(Device const& dev) const { return data_.get(alpaka::getNativeHandle(dev)); }

    DeviceObject const& get(Queue const& queue) const { return get(alpaka::getDev(queue)); }

  private:
    Impl data_;
  };
}  // namespace cms::alpakatools

#endif

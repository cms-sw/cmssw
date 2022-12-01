#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceProduct_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceProduct_h

#include <memory>
#include <optional>
#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The sole purpose of this wrapper class is to segregate the
   * EventSetup products in the device memory from the host memory
   *
   * In contrast to ED side, no synchronization are needed here as we
   * mark the ES product done only after all the (in the future
   * asynchronous) work has finished.
   */
  template <typename T>
  class ESDeviceProduct {
  public:
    virtual ~ESDeviceProduct() {}

    T const& get(Device const& dev) const { return *cache_[alpaka::getNativeHandle(dev)]; }

  protected:
    explicit ESDeviceProduct(size_t ndevices) : cache_(ndevices, nullptr) {}

    void setCache(size_t idev, T const* data) { cache_[idev] = data; }

  private:
    // trading memory to avoid virtual function
    std::vector<T const*> cache_;
  };

  namespace detail {
    /**
     * This class holds the actual storage (since EventSetup proxies
     * are able to hold std::optional<T>, std::unique_ptr<T>, and
     * std::shared_ptr<T>()). The object of this class holds the
     * storage, while the consumers of the ESProducts see only the
     * base class.
     */
    template <typename TProduct, typename TStorage>
    class ESDeviceProductWithStorage : public ESDeviceProduct<TProduct> {
      using Base = ESDeviceProduct<TProduct>;

    public:
      explicit ESDeviceProductWithStorage(size_t ndevices) : Base(ndevices), data_(ndevices) {}

      void insert(Device const& dev, TStorage data) {
        auto const idev = alpaka::getNativeHandle(dev);
        data_[idev] = std::move(data);
        this->setCache(idev, &*data_[idev]);
      }

    private:
      std::vector<TStorage> data_;
    };
  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

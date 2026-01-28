#ifndef DataFormats_Common_interface_DeviceProduct_h
#define DataFormats_Common_interface_DeviceProduct_h

#include <cassert>
#include <memory>
#include <type_traits>

#include "DataFormats/Common/interface/Uninitialized.h"

namespace edm {
  /**
   * Base class for device product metadata. The metadata object should hold device-specific information and
   * functionality necessary to synchronize the asynchronous device operations with the host, or with other asynchronous
   * operations (such as a queue used to produce another data product).
   *
   * The deriving class must implement the synchronize() virtual function that should synchronize the asynchronous
   * operations with the host. This function is called before the data product is destructed.
   *
   * The deriving class must also implement a
   * \code
   * template <typename... Args>
   * void synchronize(Args&&... args) const;
   * \endcode
   * member function that is used when the data products are accessed via DeviceProduct::getSynchronized() function. The
   * function arguments are specific to the metadata type itself and the system that makes use of it (e.g. the Alpaka
   * system).
   */
  class DeviceProductMetadataBase {
  public:
    DeviceProductMetadataBase() = default;
    virtual ~DeviceProductMetadataBase() noexcept = default;

    virtual void synchronize() const noexcept = 0;
  };

  /**
   * A wrapper for Event Data product in device memory accompanied
   * with some device-specific metadata. Not intended to be used directly by
   * developers (except in ROOT dictionary declarations in
   * classes_def.xml similar to edm::Wrapper).
   */
  template <typename T>
  class DeviceProduct {
  public:
    DeviceProduct()
      requires(requires { T(); })
    = default;

    explicit DeviceProduct(edm::Uninitialized)
      requires(requires { T(edm::kUninitialized); })
        : data_{edm::kUninitialized} {}

    template <typename M, typename... Args>
      requires std::is_base_of_v<DeviceProductMetadataBase, M>
    explicit DeviceProduct(std::shared_ptr<M> metadata, Args&&... args)
        : metadata_(std::move(metadata)), metadataType_(&typeid(M)), data_(std::forward<Args>(args)...) {}

    DeviceProduct(const DeviceProduct&) = delete;
    DeviceProduct& operator=(const DeviceProduct&) = delete;
    DeviceProduct(DeviceProduct&&) = default;
    DeviceProduct& operator=(DeviceProduct&&) = default;

    ~DeviceProduct() noexcept {
      // Ensure all asynchronous operations are synchronized before data_ is destructed
      if (metadata_) {
        metadata_->synchronize();
      }
    }

    /**
     * Get the actual data product after the metadata object has
     * synchronized the access. The synchronization details depend on
     * the metadata type, which the caller must know. All the
     * arguments are passed to M::synchronize() function.
     */
    template <typename M, typename... Args>
      requires std::is_base_of_v<DeviceProductMetadataBase, M>
    T const& getSynchronized(Args&&... args) const {
      auto const& md = metadata<M>();
      md.synchronize(std::forward<Args>(args)...);
      return data_;
    }

    // TODO: in principle this function is an implementation detail
    template <typename M>
      requires std::is_base_of_v<DeviceProductMetadataBase, M>
    M const& metadata() const {
      assert(metadata_ != nullptr);
      // TODO: I believe the assertion could be removed safely after
      // the data dependence and scheduling systems would guarantee
      // that the an EDModule in a given execution space can access
      // only to the EDProducts in a memory space compatible with the
      // execution space.
      //
      // On the other hand, with Alpaka (likely with others) the
      // getSynchronized() does additional checks so the added cost is
      // probably not that much?
      assert(typeid(M) == *metadataType_);
      return *static_cast<M const*>(metadata_.get());
    }

  private:
    std::shared_ptr<DeviceProductMetadataBase const> metadata_;  //!
    std::type_info const* metadataType_ = nullptr;               //!
    T data_;                                                     //!
  };
}  // namespace edm
#endif

#ifndef DataFormats_Portable_interface_Product_h
#define DataFormats_Portable_interface_Product_h

#include <memory>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/ProductBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/ScopedContextFwd.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace edm {
  template <typename T>
  class Wrapper;
}

namespace cms::alpakatools {

  /**
     * The purpose of this class is to wrap alpaka-based data products for
     * the edm::Event in a way which forces correct use of various utilities.
     *
     * The default constructor is needed only for the ROOT dictionary generation.
     *
     * The non-default construction has to be done with ScopedContext
     * (in order to properly register the alpaka event).
     *
     * The alpaka event is in practice needed only for inter-queue
     * synchronization, but someone with long-enough lifetime has to own
     * it. Here is a somewhat natural place. If the overhead is too much, we
     * can use them only where synchronization between queues is needed.
     */
  template <typename TQueue, typename T, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  class Product : public ProductBase<TQueue> {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;

    Product() = default;  // Needed only for ROOT dictionary generation

    Product(const Product&) = delete;
    Product& operator=(const Product&) = delete;
    Product(Product&&) = default;
    Product& operator=(Product&&) = default;

  private:
    friend class impl::ScopedContextGetterBase<Queue>;
    friend class ScopedContextProduce<Queue>;
    friend class edm::Wrapper<Product<Queue, T>>;

    explicit Product(std::shared_ptr<Queue> queue, std::shared_ptr<Event> event, T data)
        : ProductBase<Queue>(std::move(queue), std::move(event)), data_(std::move(data)) {}

    template <typename... Args>
    explicit Product(std::shared_ptr<Queue> queue, std::shared_ptr<Event> event, Args&&... args)
        : ProductBase<Queue>(std::move(queue), std::move(event)), data_(std::forward<Args>(args)...) {}

    T data_;  //!
  };

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_Product_h

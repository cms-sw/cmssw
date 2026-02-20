#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_Writer_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_Writer_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceProductType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Common.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Writer for device products.
  // T is the inner product type (e.g. PortableDeviceCollection<...>)
  template <typename T>
  class Writer : public WriterBase {
    static_assert(::ngt::HasMemoryCopyTraits<T>, "No specialization of MemoryCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<detail::DeviceProductType<T>>;

    Writer() : WriterBase(), data_{construct_()} {}

    ::ngt::AnyBuffer uninitialized_parameters() const override { return ::ngt::get_properties<T>(object()); }

    void initialize(Queue& queue, ::ngt::AnyBuffer const& args) override {
      using Traits = ::ngt::MemoryCopyTraits<T>;
      if constexpr (::ngt::HasTrivialCopyProperties<T>) {
        // properties are required to initialize an object of type T
        using Props = ::ngt::TrivialCopyProperties<T>;
        static_assert(
            requires(T& o, Queue& q, Props p) { Traits::initialize(q, o, p); },
            "MemoryCopyTraits<T> has properties but no initialize(Queue&, T&, Properties const&)");
        Traits::initialize(queue, object(), args.cast_to<Props>());
      } else {
        // No properties are required to initialize an object of type T
        static_assert(
            requires(T& o, Queue& q) { Traits::initialize(q, o); },
            "MemoryCopyTraits<T> has no properties but no initialize(Queue&, T&)");
        Traits::initialize(queue, object());
      }
    }

    std::vector<std::span<std::byte>> regions() override { return ::ngt::get_regions<T>(object()); }

    void finalize() override { ::ngt::do_finalize<T>(object()); }

    std::unique_ptr<edm::WrapperBase> get() override {
      return std::make_unique<WrapperType>(edm::WrapperBase::Emplace{}, std::move(data_));
    }

    // Extract the product T directly by move.
    T takeProduct() { return std::move(data_); }

  private:
    static T construct_() {
      if constexpr (requires { T(); }) {
        return T{};
      } else {
        return T{edm::kUninitialized};
      }
    }

    T const& object() const { return data_; }
    T& object() { return data_; }

    T data_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_Writer_h

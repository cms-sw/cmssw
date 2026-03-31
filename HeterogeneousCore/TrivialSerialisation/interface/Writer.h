#ifndef HeterogeneousCore_TrivialSerialisation_interface_Writer_h
#define HeterogeneousCore_TrivialSerialisation_interface_Writer_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Common.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"

namespace ngt {

  // Writer for host products: creates a Wrapper<T> and exposes its memory regions for writing.
  template <typename T>
  class Writer : public WriterBase {
    static_assert(ngt::HasMemoryCopyTraits<T>, "No specialization of MemoryCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<T>;

    Writer() : WriterBase() {
      // See edm::Wrapper::construct_().
      if constexpr (requires { T(); }) {
        ptr_ = std::make_unique<edm::Wrapper<T>>(edm::WrapperBase::Emplace{});
      } else {
        ptr_ = std::make_unique<edm::Wrapper<T>>(edm::WrapperBase::Emplace{}, edm::kUninitialized);
      }
    }

    ngt::AnyBuffer uninitialized_parameters() const override { return ngt::get_properties<T>(object()); }

    void initialize(ngt::AnyBuffer const& args) override {
      if constexpr (not ngt::HasValidInitialize<T>) {
        // If MemoryCopyTraits<T> has no valid initialize() then the object must be default-constructible.
        // Check that Properties are not present, as they are not needed and cannot be used.
        static_assert(not ngt::HasTrivialCopyProperties<T>);
      } else if constexpr (not ngt::HasTrivialCopyProperties<T>) {
        // If T has no TrivialCopyProperties, call initialize() without any additional arguments.
        ngt::MemoryCopyTraits<T>::initialize(object());
      } else {
        // If T has TrivialCopyProperties, cast args to Properties and pass it as an additional argument to initialize().
        ngt::MemoryCopyTraits<T>::initialize(object(), args.cast_to<ngt::TrivialCopyProperties<T>>());
      }
    }

    std::vector<std::span<std::byte>> regions() override { return ngt::get_regions<T>(object()); }

    void finalize() override { ngt::do_finalize<T>(object()); }

  private:
    const T& object() const { return static_cast<const WrapperType*>(ptr_.get())->bareProduct(); }

    T& object() { return static_cast<WrapperType*>(ptr_.get())->bareProduct(); }
  };

}  // namespace ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_Writer_h

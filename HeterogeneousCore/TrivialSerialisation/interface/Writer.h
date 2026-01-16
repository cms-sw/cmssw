#ifndef TrivialSerialisation_Common_interface_Writer_h
#define TrivialSerialisation_Common_interface_Writer_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"

namespace ngt {

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

    ngt::AnyBuffer uninitialized_parameters() const override {
      if constexpr (not ngt::HasTrivialCopyProperties<T>) {
        // if ngt::MemoryCopyTraits<T>::properties(...) is not declared, do not call it.
        return {};
      } else {
        // if ngt::MemoryCopyTraits<T>::properties(...) is declared, call it and wrap the result in an ngt::AnyBuffer
        return ngt::AnyBuffer(ngt::MemoryCopyTraits<T>::properties(object()));
      }
    }

    void initialize(ngt::AnyBuffer const& args) override {
      if constexpr (not ngt::HasValidInitialize<T>) {
        // If there is no valid initialize(), this shouldn't be present.
        static_assert(not ngt::HasTrivialCopyProperties<T>);
      } else if constexpr (not ngt::HasTrivialCopyProperties<T>) {
        // If T has no TrivialCopyProperties, call initialize() without any additional arguments.
        ngt::MemoryCopyTraits<T>::initialize(object());
      } else {
        // If T has TrivialCopyProperties, cast args to Properties and pass it as an additional argument to initialize().
        ngt::MemoryCopyTraits<T>::initialize(object(), args.cast_to<ngt::TrivialCopyProperties<T>>());
      }
    }

    std::vector<std::span<std::byte>> regions() override {
      static_assert(ngt::HasRegions<T>);
      return ngt::MemoryCopyTraits<T>::regions(object());
    }

    void finalize() override {
      if constexpr (ngt::HasTrivialCopyFinalize<T>) {
        ngt::MemoryCopyTraits<T>::finalize(object());
      }
    }

  private:
    const T& object() const { return static_cast<const WrapperType*>(ptr_.get())->bareProduct(); }

    T& object() { return static_cast<WrapperType*>(ptr_.get())->bareProduct(); }
  };

}  // namespace ngt

#endif  // TrivialSerialisation_Common_interface_Writer_h

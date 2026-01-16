#ifndef TrivialSerialisation_Common_interface_Reader_h
#define TrivialSerialisation_Common_interface_Reader_h

#include <vector>
#include <span>
#include <cstddef>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"

namespace ngt {

  template <typename T>
  class Reader : public ReaderBase {
    static_assert(ngt::HasMemoryCopyTraits<T>, "No specialization of MemoryCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<T>;

    Reader(WrapperType const& wrapper) : ReaderBase(&wrapper) {}

    ngt::AnyBuffer parameters() const override {
      if constexpr (not ngt::HasTrivialCopyProperties<T>) {
        // if ngt::MemoryCopyTraits<T>::properties(...) is not declared, do not call it.
        return {};
      } else {
        // if ngt::MemoryCopyTraits<T>::properties(...) is declared, call it and wrap the result in an ngt::AnyBuffer
        return ngt::AnyBuffer(ngt::MemoryCopyTraits<T>::properties(object()));
      }
    }

    std::vector<std::span<const std::byte>> regions() const override {
      static_assert(ngt::HasRegions<T>);
      return ngt::MemoryCopyTraits<T>::regions(object());
    }

  private:
    const T& object() const {
      const WrapperType& w = static_cast<const WrapperType&>(*ptr_);
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      return w.bareProduct();
    }
  };

}  // namespace ngt

#endif  // TrivialSerialisation_Common_interface_Reader_h

#ifndef HeterogeneousCore_TrivialSerialisation_interface_Reader_h
#define HeterogeneousCore_TrivialSerialisation_interface_Reader_h

#include <vector>
#include <span>
#include <cstddef>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Common.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"

namespace ngt {

  // Reader for host products: extracts properties and memory regions from a Wrapper<T>.
  template <typename T>
  class Reader : public ReaderBase {
    static_assert(ngt::HasMemoryCopyTraits<T>, "No specialization of MemoryCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<T>;

    explicit Reader(WrapperType const& wrapper) : ReaderBase(&wrapper) {}

    ngt::AnyBuffer parameters() const override { return ngt::get_properties<T>(object()); }

    std::vector<std::span<const std::byte>> regions() const override { return ngt::get_regions<T>(object()); }

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

#endif  // HeterogeneousCore_TrivialSerialisation_interface_Reader_h

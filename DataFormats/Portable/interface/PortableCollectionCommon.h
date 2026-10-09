#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <format>
#include <limits>
#include <stdexcept>
#include <typeinfo>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/TypeDemangler.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace portablecollection {

  template <std::size_t I = 0, typename TQueue, typename Descriptor, typename ConstDescriptor>
    requires requires { Descriptor::num_cols; }
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::num_cols) {
      assert(std::get<I>(dest.buff).size_bytes() == std::get<I>(src.buff).size_bytes());
      alpaka::memcpy(
          queue,
          alpaka::createView(alpaka::getDev(queue), std::get<I>(dest.buff).data(), std::get<I>(dest.buff).size()),
          alpaka::createView(alpaka::getDev(queue), std::get<I>(src.buff).data(), std::get<I>(src.buff).size()));
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  // Helper function implementing the recursive deep copy for blocks
  template <std::size_t I = 0, typename TQueue, typename Descriptor, typename ConstDescriptor>
    requires requires { Descriptor::blocksNumber; }
  void deepCopy(TQueue& queue, Descriptor& dest, ConstDescriptor const& src) {
    if constexpr (I < ConstDescriptor::blocksNumber) {
      deepCopy(queue, std::get<I>(dest.buff), std::get<I>(src.buff));
      deepCopy<I + 1>(queue, dest, src);
    }
  }

  // Kernel for transposing layouts between SoA and AoS
  struct Transpose {
    template <alpaka::concepts::Acc TAcc, typename DstView, typename SrcView>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const TAcc& acc,
                                                   DstView destView,
                                                   const SrcView& sourceView,
                                                   const int n) const {
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, n)) {
        destView.transpose(sourceView, local_idx);
      }
    }
  };

  template <alpaka::concepts::Acc TAcc, typename TQueue, typename DstView, typename SrcView, std::integral Int>
    requires(alpaka::isQueue<TQueue>)
  ALPAKA_FN_HOST ALPAKA_FN_INLINE void transpose(TQueue& queue, DstView& dstView, const SrcView& srcView, const Int n) {
    constexpr uint32_t BlockSize = 256;
    auto const workDiv = cms::alpakatools::make_workdiv<TAcc>(
        cms::alpakatools::divide_up_by(static_cast<alpaka_common::Idx>(n), BlockSize), BlockSize);

    alpaka::exec<TAcc>(queue, workDiv, portablecollection::Transpose{}, dstView, srcView, static_cast<int>(n));
  }

  template <std::integral Int>
  constexpr int size_cast(Int input) {
    if ((std::is_signed_v<Int> && input < 0) || input > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          std::format("Invalid input value for size of PortableCollection: cannot be narrowed to positive int32. "
                      "Source type: {}, value: {} ",
                      edm::typeDemangle(typeid(Int).name()),
                      input));
    }
    return static_cast<int>(input);
  }

}  // namespace portablecollection

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h

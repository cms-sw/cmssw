#ifndef DataFormats_Portable_interface_alpaka_Transpose_h
#define DataFormats_Portable_interface_alpaka_Transpose_h

#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollectionCommon.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace detail {
    // Kernel for transposing layouts between SoA/AoS and AoS/SoA
    struct Transpose {
      template <alpaka::concepts::Acc TAcc, typename DstView, typename SrcView>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const TAcc& acc,
                                                     DstView destView,
                                                     const SrcView sourceView,
                                                     const alpaka_common::Idx n) const {
        // copy all scalar values once with one thread
        if (cms::alpakatools::once_per_grid(acc)) {
          destView.transposeScalars(sourceView);
        }
        // transpose remaining columns between data layouts
        for (auto local_idx : cms::alpakatools::uniform_elements(acc, n)) {
          destView.transposeColumns(sourceView, local_idx);
        }
      }
    };

    template <alpaka::concepts::Queue TQueue, typename DstView, typename SrcView>
    ALPAKA_FN_HOST ALPAKA_FN_INLINE void transpose(TQueue& queue,
                                                   DstView& dstView,
                                                   const SrcView& srcView,
                                                   alpaka_common::Idx n) {
      constexpr uint32_t blocksize = 256;
      const auto nBlocks = cms::alpakatools::divide_up_by(n, blocksize);
      // Use at least one block, even if there are 0 elements to make sure that scalars are copied!
      auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(nBlocks == 0 ? 1 : nBlocks, blocksize);

      alpaka::exec<Acc1D>(queue, workDiv, Transpose{}, dstView, srcView, n);
    }
  }  // namespace detail

  // Transpose the data from a SoA view into an AoS collection.
  // Requires that one of the two collections is SoA and the other is AoS,
  // and that the AoS type is the AoSWrapper of the SoA type.
  template <alpaka::concepts::Queue TQueue, typename DstCollection, typename SrcCollection>
    void transpose(TQueue& queue, DstCollection& dstCollection, const SrcCollection& srcCollection)
      requires std::is_same_v<typename DstCollection::Device, typename SrcCollection::Device> &&
              (SrcCollection::Layout::isSoA != DstCollection::Layout::isSoA) &&
              (std::is_same_v<typename SrcCollection::Layout, typename DstCollection::Layout::AoSWrapper> ||
                std::is_same_v<typename SrcCollection::Layout::AoSWrapper, typename DstCollection::Layout>) &&
              (requires { SrcCollection::Layout::blocksNumber; } && requires { DstCollection::Layout::blocksNumber; })
    {
    using DstLayout = typename DstCollection::Layout;
    using SrcLayout = typename SrcCollection::Layout;
    static_assert(SrcLayout::blocksNumber == DstLayout::blocksNumber,
                  "PortableDeviceCollection::transpose: Source and target have different number of SoABlocks");

    const auto dstSize = dstCollection.size();
    const auto srcSize = srcCollection.size();
    auto maxSize = 0;
    for (auto i = 0; i < SrcLayout::blocksNumber; ++i) {
      if (dstSize[i] != srcSize[i]) {
        ::portablecollection::throwException("PortableDeviceCollection",
                                             "transpose: size mismatch between SoA and AoS");
      }
      maxSize = std::max(maxSize, dstSize[i]);
    }
    detail::transpose(queue, dstCollection.view(), srcCollection.const_view(), maxSize);
  }

  template <alpaka::concepts::Queue TQueue, typename DstCollection, typename SrcCollection>
    void transpose(TQueue& queue, DstCollection& dstCollection, const SrcCollection& srcCollection)
      requires std::is_same_v<typename DstCollection::Device, typename SrcCollection::Device> &&
              (SrcCollection::Layout::isSoA != DstCollection::Layout::isSoA) &&
              (std::is_same_v<typename SrcCollection::Layout, typename DstCollection::Layout::AoSWrapper> ||
                std::is_same_v<typename SrcCollection::Layout::AoSWrapper, typename DstCollection::Layout>) &&
              (!requires { SrcCollection::Layout::blocksNumber; } && !requires { DstCollection::Layout::blocksNumber; })
    {
    if (dstCollection.size() != srcCollection.size()) {
      ::portablecollection::throwException("PortableDeviceCollection", "transpose: size mismatch between SoA and AoS");
    }
    detail::transpose(queue, dstCollection.view(), srcCollection.const_view(), dstCollection.size());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_Portable_interface_alpaka_Transpose_h

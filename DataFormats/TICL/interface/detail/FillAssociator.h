
#ifndef DataFormats_TICL_interface_detail_FillAssociator_h
#define DataFormats_TICL_interface_detail_FillAssociator_h

#include "DataFormats/TICL/interface/AssociationMap.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <span>

namespace ticl::associator::detail {

  struct KernelComputeAssociationSizes {
    template <alpaka::concepts::Acc TAcc, std::integral TKey>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  std::span<const TKey> keys,
                                  TKey* keys_counts,
                                  std::size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        alpaka::atomicAdd(acc, &keys_counts[keys[i]], TKey{1});
      }
    }
  };

  struct KernelFillAssociator {
    template <alpaka::concepts::Acc TAcc, std::integral TKey, concepts::trivially_copyable TMapped>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  ticl::AssociationMapView<TKey, TMapped> view,
                                  std::span<const TKey> keys,
                                  std::span<const TMapped> values,
                                  TKey* temp_offsets) const {
      for (auto i : alpaka::uniformElements(acc, values.size())) {
        const auto key = keys[i];
        const auto offset = alpaka::atomicAdd(acc, &temp_offsets[key], TKey{1});
        view.content().values()[offset] = values[i];
      }
    }
  };

  template <alpaka::concepts::Acc TAcc, typename TQueue, std::integral TKey, concepts::trivially_copyable TMapped>
    requires alpaka::isQueue<TQueue>
  ALPAKA_FN_HOST auto fill(TQueue& queue,
                           ticl::AssociationMapView<TKey, TMapped>& map,
                           std::span<const TKey> keys,
                           std::span<const TMapped> values) {
    using namespace ::cms::alpakatools;

    const auto nkeys = map.metadata().size()[1];
    const auto nvalues = map.metadata().size()[0];

    const auto blocksize = 1024u;
    const auto gridsize = divide_up_by(keys.size(), blocksize);
    const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);
    auto keys_counts = make_device_buffer<TKey[]>(queue, nkeys);
    alpaka::memset(queue, keys_counts, 0);
    alpaka::exec<TAcc>(queue, workdiv, KernelComputeAssociationSizes{}, keys, keys_counts.data(), nvalues);

    // prepare for prefix scan
    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);
    auto temp_offsets = make_device_buffer<TKey[]>(queue, nkeys + 1);
    alpaka::memset(queue, temp_offsets, 0);
    const auto blocksize_multiblockscan = 1024;
    auto gridsize_multiblockscan = divide_up_by(nkeys, blocksize_multiblockscan);
    const auto workdiv_multiblockscan = make_workdiv<TAcc>(gridsize_multiblockscan, blocksize_multiblockscan);
    auto warp_size = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
    alpaka::exec<TAcc>(queue,
                       workdiv_multiblockscan,
                       multiBlockPrefixScan<TKey>{},
                       keys_counts.data(),
                       temp_offsets.data() + 1,
                       nkeys,
                       gridsize_multiblockscan,
                       block_counter.data(),
                       warp_size);

    alpaka::memcpy(queue,
                   make_device_view(queue, map.offsets().keys_offsets().data(), nkeys),
                   make_device_view(queue, temp_offsets.data() + 1, nkeys));
    alpaka::exec<TAcc>(queue, workdiv, KernelFillAssociator{}, map, keys, values, temp_offsets.data());
  }

}  // namespace ticl::associator::detail

#endif

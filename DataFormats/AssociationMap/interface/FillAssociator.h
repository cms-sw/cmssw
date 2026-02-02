
#ifndef DataFormats_AssociationMap_interface_FillAssociator_h
#define DataFormats_AssociationMap_interface_FillAssociator_h

#include "DataFormats/AssociationMap/interface/AssociationMap.h"
#include "DataFormats/AssociationMap/interface/detail/FillAssociator.h"
#include <alpaka/alpaka.hpp>
#include <span>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace associator {

    template <typename TAcc,
              typename TQueue,
              ticl::concepts::trivially_copyable TMapped = uint32_t,
              std::integral TKey = uint32_t>
      requires alpaka::isQueue<TQueue> && alpaka::isAccelerator<TAcc>
    ALPAKA_FN_HOST auto fill_associator(TQueue& queue,
                                        ticl::AssociationMapView<TMapped, TKey>& map,
                                        std::span<const TKey> keys,
                                        std::span<const TMapped> values) {
      detail::fill_associator<TAcc>(queue, map, keys, values);
    }

  }  // namespace associator
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

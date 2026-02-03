
#ifndef DataFormats_AssociationMap_interface_FillAssociator_h
#define DataFormats_AssociationMap_interface_FillAssociator_h

#include "DataFormats/AssociationMap/interface/AssociationMap.h"
#include "DataFormats/AssociationMap/interface/detail/FillAssociator.h"
#include <alpaka/alpaka.hpp>
#include <span>

namespace ticl::associator {

  template <alpaka::concepts::Acc TAcc,
            typename TQueue,
            ticl::concepts::trivially_copyable TMapped = uint32_t,
            std::integral TKey = uint32_t>
    requires alpaka::isQueue<TQueue>
  ALPAKA_FN_HOST auto fill(TQueue& queue,
                           ticl::AssociationMapView<TMapped, TKey>& map,
                           std::span<const TKey> keys,
                           std::span<const TMapped> values) {
    detail::fill<TAcc>(queue, map, keys, values);
  }

}  // namespace ticl::associator

#endif

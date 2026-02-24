
#ifndef DataFormats_TICL_interface_AssociationMap_h
#define DataFormats_TICL_interface_AssociationMap_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace ticl {

  namespace concepts {

    template <typename T>
    concept trivially_copyable = std::is_trivially_copyable_v<T>;

  }

  // clang-format off
  template <concepts::trivially_copyable TMapped, std::integral TKey>
  struct AssociationMapLayout {
    GENERATE_SOA_LAYOUT(ContentBuffersLayout, SOA_COLUMN(TMapped, values))
    GENERATE_SOA_LAYOUT(OffsetBufferLayout, SOA_COLUMN(TKey, keys_offsets))

    GENERATE_SOA_BLOCKS(Layout,
                        SOA_BLOCK(content, ContentBuffersLayout),
                        SOA_BLOCK(offsets, OffsetBufferLayout),
                        SOA_VIEW_METHODS(
							constexpr SOA_HOST_DEVICE auto operator[](TKey key) {
							  auto offset = (key == 0u) ? 0u : this->offsets()[key].keys_offsets();
							  auto size = (key == 0u) ? this->offsets()[0].keys_offsets() 
                                        : this->offsets()[key].keys_offsets() - this->offsets()[key - 1].keys_offsets();
							  return std::span<TMapped>{this->content().values().data() + offset, size};
							}
						),
                        SOA_CONST_VIEW_METHODS(
							constexpr SOA_HOST_DEVICE auto operator[](TKey key) const {
							  auto offset = (key == 0) ? 0u : this->offsets().keys_offsets()[key];
							  auto size = (key == 0u) ? this->offsets()[0].keys_offsets()
                                        : this->offsets()[key].keys_offsets() - this->offsets()[key - 1].keys_offsets();
							  return std::span<const TMapped>{this->content().values().data() + offset, size};
							}
							constexpr SOA_HOST_DEVICE auto contains(TKey key) const {
							  return (key == 0u) ? this->offsets()[0].keys_offsets() > 0 
                                   : this->offsets()[key].keys_offsets() > this->offsets()[key - 1].keys_offsets();
							}
							constexpr SOA_HOST_DEVICE auto count(TKey key) const {
							  return (key == 0u) ? this->offsets()[0].keys_offsets()
                                   : this->offsets()[key].keys_offsets() - this->offsets()[key - 1].keys_offsets();
							}
							constexpr SOA_HOST_DEVICE auto keys() const {
							  return this->offsets().metadata().size();
							}
							constexpr SOA_HOST_DEVICE auto size() const {
							  return this->offsets().metadata().size();
							}
						)
    )
  };
  // clang-format on

  template <concepts::trivially_copyable TMapped = uint32_t, std::integral TKey = uint32_t>
  using AssociationMap = typename AssociationMapLayout<TMapped, TKey>::template Layout<>;
  template <concepts::trivially_copyable TMapped = uint32_t, std::integral TKey = uint32_t>
  using AssociationMapView = typename AssociationMap<TMapped, TKey>::View;
  template <concepts::trivially_copyable TMapped = uint32_t, std::integral TKey = uint32_t>
  using AssociationMapConstView = typename AssociationMap<TMapped, TKey>::ConstView;

}  // namespace ticl

#endif

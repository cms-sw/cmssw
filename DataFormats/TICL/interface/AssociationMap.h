
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
  template <std::integral TKey, concepts::trivially_copyable TMapped>
  struct AssociationMapLayout {
    GENERATE_SOA_LAYOUT(ContentBuffersLayout, SOA_COLUMN(TMapped, values))
    GENERATE_SOA_LAYOUT(OffsetBufferLayout, SOA_COLUMN(TKey, keys_offsets))
    
    template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
              bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed>
    class OffsetsLayout final : public OffsetBufferLayout<ALIGNMENT, ALIGNMENT_ENFORCEMENT> {
      using Parent = OffsetBufferLayout<ALIGNMENT, ALIGNMENT_ENFORCEMENT>;

    public:
      OffsetsLayout() : Parent() {}
      OffsetsLayout(std::byte* mem, cms::soa::size_type elements) : Parent(mem, elements + 1) {}

      static constexpr auto computeDataSize(std::size_t size) {
        return Parent::computeDataSize(size + 1);
      }
    };

    using OffsetsView = OffsetBufferLayout<>::template ViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::disabled>;
    using OffsetsConstView = OffsetBufferLayout<>::template ConstViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::disabled>;

    GENERATE_SOA_BLOCKS(Layout,
                        SOA_BLOCK(content, ContentBuffersLayout),
                        SOA_BLOCK(offsets, OffsetsLayout),
                        SOA_VIEW_METHODS(
							constexpr SOA_HOST_DEVICE auto operator[](TKey key) {
                auto view = OffsetsView(this->offsets().metadata().size(), this->offsets().keys_offsets().data());

							 auto offset = view[key].keys_offsets();
							 auto size = this->count(key);
							 return std::span<TMapped>{this->content().values().data() + offset, static_cast<std::size_t>(size)};
							}
						),
                        SOA_CONST_VIEW_METHODS(
							constexpr SOA_HOST_DEVICE auto operator[](TKey key) const {
                auto view = OffsetsConstView(this->offsets().metadata().size(), this->offsets().keys_offsets().data());

							 auto offset = view[key].keys_offsets();
							 auto size = this->count(key);
							 return std::span<const TMapped>{this->content().values().data() + offset, static_cast<std::size_t>(size)};
							}
							constexpr SOA_HOST_DEVICE auto contains(TKey key) const {
							return this->count(key) > 0;
							}
							constexpr SOA_HOST_DEVICE auto count(TKey key) const {
                auto view = OffsetsConstView(this->offsets().metadata().size(), this->offsets().keys_offsets().data());
							return view[key + 1].keys_offsets() - view[key].keys_offsets();
							}
							constexpr SOA_HOST_DEVICE auto keys() const {
							return this->offsets().metadata().size() - 1;
							}
							constexpr SOA_HOST_DEVICE auto size() const {
							return this->content().metadata().size();
							}
						)
    )
  };
  // clang-format on

  template <std::integral TKey = uint32_t, concepts::trivially_copyable TMapped = uint32_t>
  using AssociationMap = typename AssociationMapLayout<TKey, TMapped>::template Layout<>;
  template <std::integral TKey = uint32_t, concepts::trivially_copyable TMapped = uint32_t>
  using AssociationMapView = typename AssociationMap<TKey, TMapped>::View;
  template <std::integral TKey = uint32_t, concepts::trivially_copyable TMapped = uint32_t>
  using AssociationMapConstView = typename AssociationMap<TKey, TMapped>::ConstView;

}  // namespace ticl

#endif

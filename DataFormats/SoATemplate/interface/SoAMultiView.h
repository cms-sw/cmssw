#ifndef DataFormats_Portable_interface_SoAMultiView_h
#define DataFormats_Portable_interface_SoAMultiView_h

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <ranges>
#include <span>

#include "SoACommon.h"

/**
 * @brief Aggregates multiple ConstViews into a single combined view.
 *
 * `SoAMultiView` stores multiple ConstViews within an `std::array`, 
 * accompanied by an offset array to enable access via a global index. 
 * An `SoAMultiView` can be passed directly by value to kernels without 
 * requiring device memory copies.
 *
 * Since the underlying SoA memories are not contiguous, cacheline inefficiencies 
 * may arise. Therefore, `SoAMultiView` is best suited for use with ConstViews, 
 * when the underlying buffer is large enough to amortize the overhead of 
 * non-contiguous access patterns when iterating over all elements.
 *
 */

template <typename ConstView, cms::soa::size_type MaxSize>
class SoAMultiView {
public:
  using ConstElement = typename ConstView::const_element;
  using size_type = cms::soa::size_type;
  constexpr static cms::soa::RangeChecking::Mode rangeChecking = ConstView::rangeChecking;

  SoAMultiView() = default;

  template <std::ranges::input_range Collections, typename Getter>
    requires std::invocable<Getter&, std::ranges::range_reference_t<Collections>>
  explicit SoAMultiView(Collections const& collections, Getter getter) {
    for (const auto& collection : collections) {
      ConstView view = std::invoke(getter, collection);
      addView(view, view.metadata().size());
    }
  }

  template <std::ranges::input_range Collections, typename Getter>
    requires std::invocable<Getter&, std::ranges::range_reference_t<Collections>>
  explicit SoAMultiView(Collections const& collections, Getter getter, std::span<const size_type> sizes) {
    for (const auto& collection : collections) {
      assert(n_ < static_cast<size_type>(sizes.size()) && "More collections provided than sizes");
      addView(std::invoke(getter, collection), sizes[n_]);
    }
  }

  void addView(ConstView view, const size_type size) {
    assert(n_ < MaxSize && "Exceeded maximum number of views");
    assert(size <= static_cast<size_type>(view.metadata().size()) && "Provided size exceeds elements in the view");

    views_[n_] = view;
    totalSize_ += size;
    offsets_[n_] = totalSize_;
    n_++;
  }

  [[nodiscard]] SOA_HOST_DEVICE SOA_INLINE ConstElement
  operator[](cms::soa::detail::IndexWithSourceLocation<rangeChecking> globalIndex) const {
    if constexpr (rangeChecking != cms::soa::RangeChecking::disabled) {
      if (globalIndex.value_ >= totalSize_ or globalIndex.value_ < 0) {
        SOA_THROW_OUT_OF_RANGE("Index surpasses size total size of SoAMultiView", globalIndex, size())
      }
    }

    if constexpr (MaxSize == 1) {
      return views_[0][globalIndex.value_];
    } else {
      if (globalIndex.value_ < offsets_[0]) {
        return views_[0][globalIndex.value_];
      }

      const size_type viewIdx = viewIndex(globalIndex.value_);
      return views_[viewIdx][globalIndex.value_ - offsets_[viewIdx - 1]];
    }
  }

  [[nodiscard]] SOA_HOST_DEVICE SOA_INLINE ConstView
  view(cms::soa::detail::IndexWithSourceLocation<rangeChecking> i) const {
    if constexpr (rangeChecking != cms::soa::RangeChecking::disabled) {
      if (i.value_ >= n_) {
        SOA_THROW_OUT_OF_RANGE("Out of range index in SoAMultiView::view()", i, n_)
      }
    }
    return views_[i.value_];
  }

  [[nodiscard]] SOA_HOST_DEVICE SOA_INLINE size_type size() const { return totalSize_; }
  [[nodiscard]] SOA_HOST_DEVICE SOA_INLINE size_type numViews() const { return n_; }

private:
  SOA_HOST_DEVICE SOA_INLINE size_type viewIndex(const size_type globalIndex) const {
    size_type viewIdx = 1;
    for (size_type i = 1; i < MaxSize; ++i) {
      viewIdx += static_cast<size_type>(n_ > i && globalIndex < offsets_[i - 1]);
    }
    return viewIdx;
  }

  std::array<ConstView, MaxSize> views_{};
  std::array<size_type, MaxSize> offsets_{};

  size_type n_{0};
  size_type totalSize_{0};
};

#endif  // DataFormats_Portable_interface_SoAMultiView_h

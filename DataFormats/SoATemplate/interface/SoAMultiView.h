#ifndef DataFormats_Portable_interface_SoAMultiView_h
#define DataFormats_Portable_interface_SoAMultiView_h

#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "SoACommon.h"

/**
 * @brief Aggregates multiple views into a single combined view, similar to `MultiVectorManager`.
 *
 * `SoAMultiView` stores multiple views as references within an `std::array`, 
 * accompanied by an offset array to enable access via a global index. 
 * Thanks to the use of `std::array`, instances of `SoAMultiView` can be passed 
 * directly by value to kernels without requiring device memory copies.
 *
 * This manager does not own or copy the underlying data; instead, it maintains lightweight 
 * references to existing views, minimizing memory overhead and construction cost.
 * However, since the underlying SoA memories are not contiguous, cacheline inefficiencies 
 * may arise. Therefore, `SoAMultiView` is best suited for use with large SoA views 
 * where such overhead is amortized.
 *
 * To ensure clarity and performance, SoA views must be provided explicitly through the 
 * constructor—no dynamic addition of views is supported.
 */

template <typename ConstView, uint8_t MaxSize = 5>
class SoAMultiView {
public:
  using ConstElement = typename ConstView::const_element;

  SoAMultiView() = default;

  // TODO try to get closer to MultiSpan
  template <typename Collections, typename Getter>
  explicit SoAMultiView(const Collections& collections, Getter getter) {
    size_type totalSize = 0;
    for (const auto& collection : collections) {
      assert(n_ < MaxSize && "Exceeded maximum number of views");

      views_[n_] = getter(collection);

      totalSize += static_cast<size_type>(views_[n_].metadata().size());
      offsets_[n_] = totalSize;
      n_++;
    }

    totalSize_ = offset;
  }

  template <typename Collections, typename Getter, typename Sizes>
  explicit SoAMultiView(const Collections& collections, Getter getter, const Sizes& sizes) {
    size_type totalSize = 0;
    for (const auto& collection : collections) {
      assert(n_ < MaxSize && "Exceeded maximum number of views");

      views_[n_] = getter(collection);

      assert(static_cast<size_type>(sizes[n_]) <= static_cast<size_type>(views_[n_].metadata().size()) &&
             "Provided size exceeds view metadata().size()");
      totalSize += static_cast<size_type>(sizes[n_]);
      offsets_[n_] = totalSize;
      n_++;
    }
  }

  SOA_HOST_DEVICE SOA_INLINE ConstElement operator[](const size_type globalIndex) const {
    return viewIndex<0>(globalIndex);
  }

  template <typename Func, typename ReduceOp>
  SOA_HOST_DEVICE auto getScalar(Func func, ReduceOp reduceOp) {
    auto result = func(views_[0]);

    for (std::size_t i = 1; i < n_; ++i) {
      result = reduceOp(result, func(views_[i]));
    }

    return result;
  }

  template <typename Func>
  SOA_HOST_DEVICE auto getScalar(Func func) {
    return func(views_[0]);
  }

  SOA_HOST_DEVICE SOA_INLINE ConstView view(size_type i) const {
    if (i >= n_) {
      SOA_THROW_OUT_OF_RANGE("Out of range index in SoAMultiView::view()", i, n_)
    }
    return views_[i];
  }
 
  SOA_HOST_DEVICE SOA_INLINE size_type size() const {
    return n_ == 0 ? static_cast<size_type>(0) : offsets_[n_ - 1];
  }

  SOA_HOST_DEVICE SOA_INLINE size_type numViews() const { return n_; }

private:
  template <int I>
  SOA_HOST_DEVICE SOA_INLINE size_type viewStart() const {
    if constexpr (I == 0) {
      return static_cast<size_type>(0);
    } else {
      return offsets_[I - 1];
    }
  }

  template <int I>
  SOA_HOST_DEVICE SOA_INLINE ConstElement viewIndex(const size_type globalIndex) const {
    if constexpr (I == MaxSize - 1) {
      return views_[I][globalIndex - viewStart<I>()];
    } else {
      if (globalIndex < offsets_[I]) {
        return views_[I][globalIndex - viewStart<I>()];
      }
      return viewIndex<I + 1>(globalIndex);
    }
  }

  std::array<ConstView, MaxSize> views_;
  std::array<size_type, MaxSize> offsets_;

  std::size_t n_{0};
};

#endif  // DataFormats_Portable_interface_SoAMultiView_h

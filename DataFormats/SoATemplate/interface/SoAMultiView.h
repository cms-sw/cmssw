#ifndef DataFormats_Portable_interface_SoAMultiView_h
#define DataFormats_Portable_interface_SoAMultiView_h

#include <array>
#include <cassert>
#include <cstdint>

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

template <typename ConstView, int MaxSize = 3>
class SoAMultiView {
public:
  using ConstElement = typename ConstView::const_element;
  using size_type = cms::soa::size_type;

  SoAMultiView() = default;

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
    return viewIndex<0>(globalIndex, 0);
  }

  template <typename Func, typename ReduceOp>
  SOA_HOST_DEVICE auto getScalar(Func func, ReduceOp reduceOp) {
    auto result = func(views_[0]);

    for (size_type i = 1; i < n_; ++i) {
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

  SOA_HOST_DEVICE SOA_INLINE size_type size() const { return n_ == 0 ? static_cast<size_type>(0) : offsets_[n_ - 1]; }

  SOA_HOST_DEVICE SOA_INLINE size_type numViews() const { return n_; }

private:
  template <int I>
  SOA_HOST_DEVICE SOA_INLINE ConstElement viewIndex(const size_type globalIndex, const size_type currentOffset) const {
    if constexpr (I == MaxSize - 1) {
      return views_[I][globalIndex - currentOffset];
    } else {
      if (globalIndex < offsets_[I]) {
        return views_[I][globalIndex - currentOffset];
      }
      return viewIndex<I + 1>(globalIndex, offsets_[I]);
    }
  }

  std::array<ConstView, MaxSize> views_;
  std::array<size_type, MaxSize> offsets_;

  size_type n_{0};
};

#endif  // DataFormats_Portable_interface_SoAMultiView_h

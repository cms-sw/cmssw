#ifndef DataFormats_Portable_interface_MultiView_h
#define DataFormats_Portable_interface_MultiView_h

#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


/**
 * @brief Aggregates multiple views into a single combined view, similar to `MultiVectorManager`.
 *
 * `MultiView` stores multiple views as references within an `std::array`, 
 * accompanied by an offset array to enable access via a global index. 
 * Thanks to the use of `std::array`, instances of `MultiView` can be passed 
 * directly by value to kernels without requiring device memory copies.
 *
 * This manager does not own or copy the underlying data; instead, it maintains lightweight 
 * references to existing views, minimizing memory overhead and construction cost.
 * However, since the underlying SoA memories are not contiguous, cacheline inefficiencies 
 * may arise. Therefore, `MultiView` is best suited for use with large SoA views 
 * where such overhead is amortized.
 *
 * To ensure clarity and performance, SoA views must be provided explicitly through the 
 * constructor—no dynamic addition of views is supported.
 */

template <typename ConstView, uint8_t MaxSize = 5>
class MultiView {
public:
  using ConstElement = typename ConstView::const_element;

  MultiView() = default;

  template <typename... ConstViews>
  MultiView(const ConstViews&... views) : views_{{views...}}, offsets_{} {
    static_assert(sizeof...(ConstViews) < MaxSize, "Number of arguments must not exceed the maximum number of views that can be added");
    ((offsets_[n_] = totalSize_, totalSize_ += views.metadata().size(), ++n_), ...);
  }

  void addView(ConstView const& constView) {
    assert(n_ < MaxSize && ("Added view exceeds the maximum number of views that can be added: " + std::to_string(MaxSize)).c_str());

    views_[n_] = constView;
    offsets_[n_] = totalSize_;
    totalSize_ += constView.metadata().size();
    ++n_;

  }

  const ALPAKA_FN_HOST_ACC ConstElement operator[](const std::size_t globalIndex) const {
    assert(globalIndex < totalSize_ && "Global index out of range");

    const std::size_t vi = viewIndex(globalIndex);
    const std::size_t li = globalIndex - offsets_[vi];
    return views_[vi][li];
  }

  template<typename Func, typename ReduceOp>
  ALPAKA_FN_HOST_ACC auto getScalar(Func func, ReduceOp reduceOp) {
      auto result = func(views_[0]);

      CMS_UNROLL_LOOP
      for (std::size_t i = 1; i < n_; ++i) {
          result = reduceOp(result, func(views_[i]));
      }

      return result;
  }

  template<typename Func>
  ALPAKA_FN_HOST_ACC auto getScalar(Func func) {
      return func(views_[0]);
  }

  ALPAKA_FN_HOST_ACC ConstView getView(const std::size_t globalIndex) const {
    assert(globalIndex < totalSize_ && "Global index out of range");

    const std::size_t vi = viewIndex(globalIndex);
    return views_[vi];
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE std::size_t viewIndex(const std::size_t globalIndex) const {
    std::size_t result = 0;

    CMS_UNROLL_LOOP
    for (std::size_t i = 0; i < n_; ++i) {
      result = (globalIndex >= offsets_[i]) ? i : result;
    }

    return result;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE std::size_t getLocalIndex(const std::size_t globalIndex) const {
    const std::size_t vi = viewIndex(globalIndex);
    const std::size_t li = globalIndex - offsets_[vi];

    return li;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE std::size_t size() const { return totalSize_; }

private:
  std::array<ConstView, MaxSize> views_;
  std::array<std::size_t, MaxSize> offsets_;
  std::size_t totalSize_{0};

  std::size_t n_{0};
};

#endif  // DataFormats_Portable_interface_MultiView_h

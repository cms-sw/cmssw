#ifndef HeterogeneousCore_AlpakaInterface_interface_vec_h
#define HeterogeneousCore_AlpakaInterface_interface_vec_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

namespace alpaka {

  //! \return The element-wise minimum of one or more vectors.
  ALPAKA_NO_HOST_ACC_WARNING
  template <typename TDim,
            typename TVal,
            typename... Vecs,
            typename = std::enable_if_t<(std::is_same_v<Vec<TDim, TVal>, Vecs> && ...)>>
  ALPAKA_FN_HOST_ACC constexpr auto elementwise_min(Vec<TDim, TVal> const& p, Vecs const&... qs) -> Vec<TDim, TVal> {
    Vec<TDim, TVal> r;
    if constexpr (TDim::value > 0) {
      for (typename TDim::value_type i = 0; i < TDim::value; ++i)
        r[i] = std::min({p[i], qs[i]...});
    }
    return r;
  }

  //! \return The element-wise maximum of one or more vectors.
  ALPAKA_NO_HOST_ACC_WARNING
  template <typename TDim,
            typename TVal,
            typename... Vecs,
            typename = std::enable_if_t<(std::is_same_v<Vec<TDim, TVal>, Vecs> && ...)>>
  ALPAKA_FN_HOST_ACC constexpr auto elementwise_max(Vec<TDim, TVal> const& p, Vecs const&... qs) -> Vec<TDim, TVal> {
    Vec<TDim, TVal> r;
    if constexpr (TDim::value > 0) {
      for (typename TDim::value_type i = 0; i < TDim::value; ++i)
        r[i] = std::max({p[i], qs[i]...});
    }
    return r;
  }

}  // namespace alpaka

#endif  // HeterogeneousCore_AlpakaInterface_interface_vec_h

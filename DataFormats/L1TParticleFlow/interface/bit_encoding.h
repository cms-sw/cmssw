#ifndef DATAFORMATS_L1TPARTICLEFLOW_ENCODING_H
#define DATAFORMATS_L1TPARTICLEFLOW_ENCODING_H

#include <cassert>
#include <type_traits>

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"

template <typename U, typename T>
inline void pack_into_bits(U& u, unsigned int& start, const T& data) {
  const unsigned int w = T::width;
  u(start + w - 1, start) = data(w - 1, 0);
  start += w;
}

template <typename U, typename T>
inline void unpack_from_bits(const U& u, unsigned int& start, T& data) {
  const unsigned int w = T::width;
  data(w - 1, 0) = u(start + w - 1, start);
  start += w;
}

template <typename U>
inline void pack_bool_into_bits(U& u, unsigned int& start, bool data) {
  u[start++] = data;
}

template <typename U>
inline void unpack_bool_from_bits(const U& u, unsigned int& start, bool& data) {
  data = u[start++];
}

// Enum to define different packing strategies for data encoding
// DEFAULT: Standard packing
// BARREL: Packing strategy for barrel region
// ENDCAP: Packing strategy for endcap region
enum class PackingStrategy { DEFAULT, BARREL, ENDCAP };

// Default case: Calls T::unpack()
template <typename T,
          int NB,
          PackingStrategy METHOD = PackingStrategy::DEFAULT,
          typename std::enable_if<METHOD == PackingStrategy::DEFAULT, int>::type = 0>
inline auto unpack_helper(const ap_uint<NB>& data) {
  static_assert(T::BITWIDTH <= NB, "NB Type is too small for the object");
  return T::unpack(data);
}

// Specialization for BARREL
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::BARREL, int>::type = 0>
inline auto unpack_helper(const ap_uint<NB>& data) {
  static_assert(T::BITWIDTH_BARREL <= NB, "NB Type is too small for the object");
  return T::unpack_barrel(data);
}

// Specialization for ENDCAP
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::ENDCAP, int>::type = 0>
inline auto unpack_helper(const ap_uint<NB>& data) {
  static_assert(T::BITWIDTH_ENDCAP <= NB, "NB Type is too small for the object");
  return T::unpack_endcap(data);
}

// Default case: Calls T::unpack()
template <typename T,
          int NB,
          PackingStrategy METHOD = PackingStrategy::DEFAULT,
          typename std::enable_if<METHOD == PackingStrategy::DEFAULT, int>::type = 0>
inline auto unpack_slim_helper(const ap_uint<NB>& data) {
  static_assert(T::BITWIDTH_SLIM <= NB, "NB Type is too small for the object");
  return T::unpack(data);
}

// Specialization for BARREL
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::BARREL, int>::type = 0>
inline auto unpack_slim_helper(const ap_uint<NB>& data) {
  static_assert(T::BITWIDTH_BARREL_SLIM <= NB, "NB Type is too small for the object");
  return T::unpack_barrel(data);
}

// Specialization for ENDCAP
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::ENDCAP, int>::type = 0>
inline auto unpack_slim_helper(const ap_uint<NB>& data) {
  static_assert(T::BITWIDTH_ENDCAP_SLIM <= NB, "NB Type is too small for the object");
  return T::unpack_endcap(data);
}

// Default case: Calls T::unpack()
template <typename T,
          int NB,
          PackingStrategy METHOD = PackingStrategy::DEFAULT,
          typename std::enable_if<METHOD == PackingStrategy::DEFAULT, int>::type = 0>
inline auto pack_helper(const T& obj) {
  static_assert(T::BITWIDTH <= NB, "NB Type is too small for the object");
  return obj.pack();
}

// Specialization for BARREL
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::BARREL, int>::type = 0>
inline auto pack_helper(const T& obj) {
  static_assert(T::BITWIDTH_BARREL <= NB, "NB Type is too small for the object");
  return obj.pack_barrel();
}

// Specialization for ENDCAP
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::ENDCAP, int>::type = 0>
inline auto pack_helper(const T& obj) {
  static_assert(T::BITWIDTH_ENDCAP <= NB, "NB Type is too small for the object");
  return obj.pack_endcap();
}

// Default case: Calls T::unpack()
template <typename T,
          int NB,
          PackingStrategy METHOD = PackingStrategy::DEFAULT,
          typename std::enable_if<METHOD == PackingStrategy::DEFAULT, int>::type = 0>
inline auto pack_slim_helper(const T& obj) {
  static_assert(T::BITWIDTH_SLIM <= NB, "NB Type is too small for the object");
  return obj.pack_slim();
}

// Specialization for BARREL
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::BARREL, int>::type = 0>
inline auto pack_slim_helper(const T& obj) {
  static_assert(T::BITWIDTH_BARREL_SLIM <= NB, "NB Type is too small for the object");
  return obj.pack_barrel_slim();
}

// Specialization for ENDCAP
template <typename T,
          int NB,
          PackingStrategy METHOD,
          typename std::enable_if<METHOD == PackingStrategy::ENDCAP, int>::type = 0>
inline auto pack_slim_helper(const T& obj) {
  static_assert(T::BITWIDTH_ENDCAP_SLIM <= NB, "NB Type is too small for the object");
  return obj.pack_endcap_slim();
}

template <unsigned int N, PackingStrategy METHOD = PackingStrategy::DEFAULT, unsigned int OFFS = 0, typename T, int NB>
inline void l1pf_pattern_pack(const T objs[N], ap_uint<NB> data[]) {
#ifdef __SYNTHESIS__
#pragma HLS inline
#pragma HLS inline region recursive
#endif
  for (unsigned int i = 0; i < N; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
    data[i + OFFS] = pack_helper<T, NB, METHOD>(objs[i]);
  }
}

// overlaod for default strategy
template <unsigned int N, unsigned int OFFS, typename T, int NB>
inline void l1pf_pattern_pack(const T objs[N], ap_uint<NB> data[]) {
  l1pf_pattern_pack<N, PackingStrategy::DEFAULT, OFFS, T, NB>(objs, data);
}

template <unsigned int N, PackingStrategy METHOD = PackingStrategy::DEFAULT, unsigned int OFFS = 0, typename T, int NB>
inline void l1pf_pattern_unpack(const ap_uint<NB> data[], T objs[N]) {
#ifdef __SYNTHESIS__
#pragma HLS inline
#pragma HLS inline region recursive
#endif
  for (unsigned int i = 0; i < N; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
    objs[i] = unpack_helper<T, NB, METHOD>(data[i + OFFS]);
  }
}

// overlaod for default strategy
template <unsigned int N, unsigned int OFFS, typename T, int NB>
inline void l1pf_pattern_unpack(const ap_uint<NB> data[], T objs[N]) {
  l1pf_pattern_unpack<N, PackingStrategy::DEFAULT, OFFS, T, NB>(data, objs);
}

template <unsigned int N, PackingStrategy METHOD = PackingStrategy::DEFAULT, unsigned int OFFS = 0, typename T, int NB>
inline void l1pf_pattern_pack_slim(const T objs[N], ap_uint<NB> data[]) {
#ifdef __SYNTHESIS__
#pragma HLS inline
#pragma HLS inline region recursive
#endif
  for (unsigned int i = 0; i < N; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
    data[i + OFFS] = pack_slim_helper<T, NB, METHOD>(objs[i]);
  }
}

// overlaod for default strategy
template <unsigned int N, unsigned int OFFS, typename T, int NB>
inline void l1pf_pattern_pack_slim(const T objs[N], ap_uint<NB> data[]) {
  l1pf_pattern_pack_slim<N, PackingStrategy::DEFAULT, OFFS, T, NB>(objs, data);
}

template <unsigned int N, PackingStrategy METHOD = PackingStrategy::DEFAULT, unsigned int OFFS = 0, typename T, int NB>
inline void l1pf_pattern_unpack_slim(const ap_uint<NB> data[], T objs[N]) {
#ifdef __SYNTHESIS__
#pragma HLS inline
#pragma HLS inline region recursive
#endif
  for (unsigned int i = 0; i < N; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
    objs[i] = unpack_slim_helper<T, NB, METHOD>(data[i + OFFS]);
  }
}

// overlaod for default strategy
template <unsigned int N, unsigned int OFFS, typename T, int NB>
inline void l1pf_pattern_unpack_slim(const ap_uint<NB> data[], T objs[N]) {
  l1pf_pattern_unpack_slim<N, PackingStrategy::DEFAULT, OFFS, T, NB>(data, objs);
}

#endif

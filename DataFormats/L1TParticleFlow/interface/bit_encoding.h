#ifndef DATAFORMATS_L1TPARTICLEFLOW_ENCODING_H
#define DATAFORMATS_L1TPARTICLEFLOW_ENCODING_H

#include <cassert>
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

template <unsigned int N, unsigned int OFFS = 0, typename T, int NB>
inline void l1pf_pattern_pack(const T objs[N], ap_uint<NB> data[]) {
#ifdef __SYNTHESIS__
#pragma HLS inline
#pragma HLS inline region recursive
#endif
  assert(T::BITWIDTH <= NB);
  for (unsigned int i = 0; i < N; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
    data[i + OFFS] = objs[i].pack();
  }
}

template <unsigned int N, unsigned int OFFS = 0, typename T, int NB>
inline void l1pf_pattern_unpack(const ap_uint<NB> data[], T objs[N]) {
#ifdef __SYNTHESIS__
#pragma HLS inline
#pragma HLS inline region recursive
#endif
  assert(T::BITWIDTH <= NB);
  for (unsigned int i = 0; i < N; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
    objs[i] = T::unpack(data[i + OFFS]);
  }
}

#endif

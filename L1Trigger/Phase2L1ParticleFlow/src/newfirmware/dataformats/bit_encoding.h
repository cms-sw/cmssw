#ifndef FIRMWARE_L1PF_ENCODING_H
#define FIRMWARE_L1PF_ENCODING_H

#include <cassert>

template <typename U, typename T> 
inline void _pack_into_bits(U & u, unsigned int & start, const T & data) {
    #pragma HLS inline
    const unsigned int w = T::width;
    u(start+w-1,start) = data(w-1,0);
    start += w;
}

template <typename U, typename T> 
inline void _unpack_from_bits(const U & u, unsigned int & start, T & data) {
    #pragma HLS inline
    const unsigned int w = T::width;
    data(w-1,0) = u(start+w-1,start);
    start += w;
}

template <typename U> 
inline void _pack_bool_into_bits(U & u, unsigned int & start, bool data) {
    #pragma HLS inline
    u[start++] = data;
}

template <typename U> 
inline void _unpack_bool_from_bits(const U & u, unsigned int & start, bool & data) {
    #pragma HLS inline
    data = u[start++];
}




template <unsigned int N, unsigned int OFFS=0, typename T, int NB>
inline void l1pf_pattern_pack(const T objs[N], ap_uint<NB> data[]) {
  #pragma HLS inline
  assert(T::BITWIDTH <= NB);
  for (unsigned int i = 0; i < N; ++i) {
    #pragma HLS unroll
    data[i + OFFS] = objs[i].pack();
  }
}

template <unsigned int N, unsigned int OFFS=0, typename T, int NB>
inline void l1pf_pattern_unpack(const ap_uint<NB> data[], T objs[N]) {
  #pragma HLS inline
  assert(T::BITWIDTH <= NB);
  for (unsigned int i = 0; i < N; ++i) {
    #pragma HLS unroll
    objs[i] = T::unpack(data[i + OFFS]);
  }
}

#endif

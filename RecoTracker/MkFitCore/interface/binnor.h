#ifndef RecoTracker_MkFitCore_interface_binnor_h
#define RecoTracker_MkFitCore_interface_binnor_h

#include "radix_sort.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <cmath>

namespace mkfit {

  // For all axis types:
  //--------------------
  // R - real type
  // I - bin index type
  // M and N - number of bits for fine and normal binning

  // axis_base
  //----------
  template <typename R, typename I, unsigned M, unsigned N>
  struct axis_base {
    static_assert(M >= N);

    typedef R real_t;
    typedef I index_t;

    static constexpr unsigned int c_M = M;
    static constexpr unsigned int c_N = N;
    static constexpr unsigned int c_M2N_shift = M - N;

    const R m_R_min, m_R_max;
    const R m_M_fac, m_N_fac;
    const R m_M_lbhp, m_N_lbhp;  // last bin half-posts
    const I m_last_M_bin, m_last_N_bin;

    struct I_pair {
      I begin;
      I end;

      I_pair() : begin(0), end(0) {}
      I_pair(I b, I e) : begin(b), end(e) {}
    };

    axis_base(R min, R max, unsigned int M_size, unsigned int N_size)
        : m_R_min(min),
          m_R_max(max),
          m_M_fac(M_size / (max - min)),
          m_N_fac(N_size / (max - min)),
          m_M_lbhp(max - 0.5 / m_M_fac),
          m_N_lbhp(max - 0.5 / m_N_fac),
          m_last_M_bin(M_size - 1),
          m_last_N_bin(N_size - 1) {
      // Requested number of bins must fit within the intended bit-field (declared by binnor, later).
      assert(M_size <= (1 << M));
      assert(N_size <= (1 << N));
      assert(max > min);
    }

    I from_R_to_M_bin(R r) const { return std::floor((r - m_R_min) * m_M_fac); }
    I from_R_to_N_bin(R r) const { return std::floor((r - m_R_min) * m_N_fac); }

    I from_R_to_M_bin_safe(R r) const { return r <= m_R_min ? 0 : (r >= m_M_lbhp ? m_last_M_bin : from_R_to_M_bin(r)); }
    I from_R_to_N_bin_safe(R r) const { return r <= m_R_min ? 0 : (r >= m_N_lbhp ? m_last_N_bin : from_R_to_N_bin(r)); }

    I from_M_bin_to_N_bin(I m) const { return m >> c_M2N_shift; }

    I_pair from_R_minmax_to_N_bins(R rmin, R rmax) const {
      return I_pair(from_R_to_N_bin_safe(rmin), from_R_to_N_bin_safe(rmax) + I{1});
    }

    I_pair from_R_rdr_to_N_bins(R r, R dr) const { return from_R_minmax_to_N_bins(r - dr, r + dr); }
    I next_N_bin(I bin) const { return bin + 1; }
  };

  // axis_pow2_base
  //---------------
  // Assumes the numbers of fine/normal bins are powers of 2 that are inferred directly from the number of bits.
  template <typename R, typename I, unsigned M, unsigned N>
  struct axis_pow2_base : public axis_base<R, I, M, N> {
    static constexpr unsigned int c_M_end = 1 << M;
    static constexpr unsigned int c_N_end = 1 << N;

    axis_pow2_base(R min, R max) : axis_base<R, I, M, N>(min, max, c_M_end, c_N_end) {}

    unsigned int size_of_M() const { return c_M_end; }
    unsigned int size_of_N() const { return c_N_end; }
  };

  // axis_pow2_u1
  //-------------
  // Specialization of axis_pow2 for the "U(1)" case where the coordinate is periodic with period (Rmax - Rmin).
  // In the "safe" methods below, bit masking serves as the modulo operator for out-of-range bin numbers.
  template <typename R, typename I, unsigned int M, unsigned int N>
  struct axis_pow2_u1 : public axis_pow2_base<R, I, M, N> {
    static constexpr I c_M_mask = (1 << M) - 1;
    static constexpr I c_N_mask = (1 << N) - 1;

    axis_pow2_u1(R min, R max) : axis_pow2_base<R, I, M, N>(min, max) {}

    I from_R_to_M_bin_safe(R r) const { return this->from_R_to_M_bin(r) & c_M_mask; }
    I from_R_to_N_bin_safe(R r) const { return this->from_R_to_N_bin(r) & c_N_mask; }

    typename axis_base<R, I, M, N>::I_pair from_R_minmax_to_N_bins(R rmin, R rmax) const {
      return typename axis_base<R, I, M, N>::I_pair(from_R_to_N_bin_safe(rmin),
                                                    (this->from_R_to_N_bin(rmax) + I{1}) & c_N_mask);
    }

    typename axis_base<R, I, M, N>::I_pair from_R_rdr_to_N_bins(R r, R dr) const {
      return from_R_minmax_to_N_bins(r - dr, r + dr);
    }
    I next_N_bin(I bin) const { return (bin + 1) & c_N_mask; }
  };

  // axis_pow2
  //----------
  template <typename R, typename I, unsigned int M, unsigned int N>
  struct axis_pow2 : public axis_pow2_base<R, I, M, N> {
    axis_pow2(R min, R max) : axis_pow2_base<R, I, M, N>(min, max) {}
  };

  // axis
  //-----
  template <typename R, typename I, unsigned M = 8 * sizeof(I), unsigned N = 8 * sizeof(I)>
  struct axis : public axis_base<R, I, M, N> {
    const unsigned int m_num_M_bins, m_num_N_bins;

    axis(R min, R max, unsigned n_bins)
        : axis_base<R, I, M, N>(min, max, n_bins << this->c_M2N_shift, n_bins),
          m_num_M_bins(n_bins << this->c_M2N_shift),
          m_num_N_bins(n_bins) {}

    axis(R min, R max, R bin_width) {
      R extent = max - min;
      unsigned int n_bins = std::ceil(extent / bin_width);
      R extra = (n_bins * bin_width - extent) / 2;

      axis(min - extra, max + extra, n_bins);
    }

    unsigned int size_of_M() const { return m_num_M_bins; }
    unsigned int size_of_N() const { return m_num_N_bins; }
  };

  // binnor
  //---------------
  // To build and populate bins, do the following:
  // 1. Construct two axis objects, giving numbers of bits and bins, and extents.
  // 2. Construct a binnor from the axis objects, and begin_registration on it.
  // 3. Loop register_entry (optional: _safe) over pairs of coordinates, to fill
  //    m_cons with the corresponding pairs of bin indices (B_pairs).
  // 4. Call finalize_registration, which sorts m_ranks based on m_cons, making
  //    m_ranks into an in-order map into m_cons (as well as the inputs that were
  //    used to fill it). Final counts for all the bins, as well as starting
  //    indices for the bins (within m_ranks), are computed and stored in packed
  //    form (i.e., bit-fields) in m_bins.
  // 5. m_cons can be kept to do preselection when determining search ranges.
  //    Note that additional precision on Axis2 is screened out during sorting.

  //
  // C - bin content type, to hold "bin population coordinates" in packed form (bit-fields)
  // A1, A2 - axis types
  // NB_first, NB_count - number of bits for storage of { first, count } pairs

  template <typename C,
            typename A1,
            typename A2,
            unsigned int NB_first = 8 * sizeof(C),
            unsigned int NB_count = 8 * sizeof(C)>
  struct binnor {
    static_assert(std::is_same<C, unsigned short>() || std::is_same<C, unsigned int>());
    static_assert(std::is_same<typename A1::real_t, typename A2::real_t>());
    static_assert(A1::c_M + A2::c_M <= 32);

    static constexpr unsigned int c_A1_mask = (1 << A1::c_M) - 1;
    static constexpr unsigned int c_A2_Mout_mask = ~(((1 << A2::c_M2N_shift) - 1) << A1::c_M);

    // Pair of axis bin indices packed into unsigned.
    struct B_pair {
      unsigned int packed_value;  // bin1 in A1::c_M lower bits, bin2 above

      B_pair() : packed_value(0) {}
      B_pair(unsigned int pv) : packed_value(pv) {}
      B_pair(typename A1::index_t i1, typename A2::index_t i2) : packed_value(i2 << A1::c_M | i1) {}

      typename A1::index_t bin1() const { return packed_value & c_A1_mask; }
      typename A2::index_t bin2() const { return packed_value >> A1::c_M; }

      unsigned int mask_A2_M_bins() const { return packed_value & c_A2_Mout_mask; }
    };

    // Bin content pair (bit-fields).
    struct C_pair {
      C first : NB_first;
      C count : NB_count;

      C_pair() : first(0), count(0) {}
      C_pair(C f, C c) : first(f), count(c) {}

      bool empty() const { return count == 0; }
      C begin() const { return first; }
      C end() const { return first + count; }
    };

    const A1 &m_a1;
    const A2 &m_a2;
    std::vector<B_pair> m_cons;
    std::vector<unsigned int> m_cons_masked;
    std::vector<C_pair> m_bins;
    std::vector<C> m_ranks;
    const bool m_radix_sort;
    const bool m_keep_cons;
    const bool m_do_masked;

    binnor(const A1 &a1, const A2 &a2, bool radix = true, bool keep_cons = false)
        : m_a1(a1),
          m_a2(a2),
          m_bins(m_a1.size_of_N() * m_a2.size_of_N()),
          m_radix_sort(radix),
          m_keep_cons(keep_cons),
          m_do_masked(radix || !keep_cons) {}

    // Access -- bin indices

    B_pair m_bin_to_n_bin(B_pair m_bin) {
      return {m_a1.from_M_bin_to_N_bin(m_bin.bin1()), m_a2.from_M_bin_to_N_bin(m_bin.bin2())};
    }

    B_pair get_n_bin(typename A1::index_t n1, typename A2::index_t n2) const { return {n1, n2}; }

    B_pair get_n_bin(typename A1::real_t r1, typename A2::real_t r2) const {
      return {m_a1.from_R_to_N_bin(r1), m_a2.from_R_to_N_bin(r2)};
    }

    // Access -- content of bins

    C_pair &ref_content(B_pair n_bin) { return m_bins[n_bin.bin2() * m_a1.size_of_N() + n_bin.bin1()]; }

    C_pair get_content(B_pair n_bin) const { return m_bins[n_bin.bin2() * m_a1.size_of_N() + n_bin.bin1()]; }

    C_pair get_content(typename A1::index_t n1, typename A2::index_t n2) const {
      return m_bins[n2 * m_a1.size_of_N() + n1];
    }

    C_pair get_content(typename A1::real_t r1, typename A2::real_t r2) const {
      return get_content(m_a1.from_R_to_N_bin(r1), m_a2.from_R_to_N_bin(r2));
    }

    // Filling

    void reset_contents(bool shrink_vectors = true) {
      if (m_keep_cons) {
        m_cons.clear();
        if (shrink_vectors)
          m_cons.shrink_to_fit();
      }
      m_bins.assign(m_bins.size(), C_pair());
      m_ranks.clear();
      if (shrink_vectors)
        m_ranks.shrink_to_fit();
    }

    void begin_registration(C n_items) {
      if (m_keep_cons)
        m_cons.reserve(n_items);
      if (m_do_masked)
        m_cons_masked.reserve(n_items);
    }

    void register_entry(B_pair bp) {
      if (m_keep_cons)
        m_cons.push_back(bp);
      if (m_do_masked)
        m_cons_masked.push_back(bp.mask_A2_M_bins());
    }

    void register_entry(typename A1::real_t r1, typename A2::real_t r2) {
      register_entry({m_a1.from_R_to_M_bin(r1), m_a2.from_R_to_M_bin(r2)});
    }

    void register_entry_safe(typename A1::real_t r1, typename A2::real_t r2) {
      register_entry({m_a1.from_R_to_M_bin_safe(r1), m_a2.from_R_to_M_bin_safe(r2)});
    }

    // Do M-binning outside, potentially using R_to_M_bin_safe().
    void register_m_bins(typename A1::index_t m1, typename A2::index_t m2) { register_entry({m1, m2}); }

    void finalize_registration() {
      unsigned int n_entries = m_do_masked ? m_cons_masked.size() : m_cons.size();
      if (m_radix_sort && n_entries >= (m_keep_cons ? 128 : 256)) {
        radix_sort<unsigned int, C> radix;
        radix.sort(m_cons_masked, m_ranks);
      } else {
        m_ranks.resize(n_entries);
        std::iota(m_ranks.begin(), m_ranks.end(), 0);
        if (m_do_masked)
          std::sort(
              m_ranks.begin(), m_ranks.end(), [&](auto &a, auto &b) { return m_cons_masked[a] < m_cons_masked[b]; });
        else
          std::sort(m_ranks.begin(), m_ranks.end(), [&](auto &a, auto &b) {
            return m_cons[a].mask_A2_M_bins() < m_cons[b].mask_A2_M_bins();
          });
      }

      for (C i = 0; i < m_ranks.size(); ++i) {
        C j = m_ranks[i];
        C_pair &c_bin = ref_content(m_bin_to_n_bin(m_keep_cons ? m_cons[j] : B_pair(m_cons_masked[j])));
        if (c_bin.count == 0)
          c_bin.first = i;
        ++c_bin.count;

#ifdef DEBUG
        B_pair n_pair = m_bin_to_n_bin(m_cons[j]);
        printf("i=%4u j=%4u  %u %u %u %u\n", i, j, n_pair.bin1, n_pair.bin2, c_bin.first, c_bin.count);
#endif
      }

      if (m_keep_cons)
        m_cons.shrink_to_fit();
      if (m_do_masked) {
        m_cons_masked.clear();
        m_cons_masked.shrink_to_fit();
      }
    }
  };

}  // namespace mkfit

#endif

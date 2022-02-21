#ifndef RecoTracker_MkFitCore_interface_binnor_h
#define RecoTracker_MkFitCore_interface_binnor_h

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <cstdio>

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

    static constexpr unsigned c_M = M;
    static constexpr unsigned c_N = N;
    static constexpr unsigned c_M2N_shift = M - N;

    const R m_R_min, m_R_max;
    const R m_M_fac, m_N_fac;
    const I m_last_M_bin, m_last_N_bin;

    struct I_pair {
      I begin;
      I end;

      I_pair() : begin(0), end(0) {}
      I_pair(I b, I e) : begin(b), end(e) {}
    };

    axis_base(R min, R max, unsigned M_size, unsigned N_size)
        : m_R_min(min),
          m_R_max(max),
          m_M_fac(M_size / (max - min)),
          m_N_fac(N_size / (max - min)),
          m_last_M_bin(M_size - 1),
          m_last_N_bin(N_size - 1) {}

    I R_to_M_bin(R r) const { return (r - m_R_min) * m_M_fac; }
    I R_to_N_bin(R r) const { return (r - m_R_min) * m_N_fac; }

    I R_to_M_bin_safe(R r) const { return r <= m_R_min ? 0 : (r >= m_R_max ? m_last_M_bin : R_to_M_bin(r)); }
    I R_to_N_bin_safe(R r) const { return r <= m_R_min ? 0 : (r >= m_R_max ? m_last_N_bin : R_to_N_bin(r)); }

    I M_bin_to_N_bin(I m) const { return m >> c_M2N_shift; }

    I_pair Rminmax_to_N_bins(R rmin, R rmax) const {
      return I_pair(R_to_N_bin_safe(rmin), R_to_N_bin_safe(rmax) + I{1});
    }

    I_pair Rrdr_to_N_bins(R r, R dr) const { return Rminmax_to_N_bins(r - dr, r + dr); }
    I next_N_bin(I bin) const { return bin + 1; }
  };

  // axis_pow2_base
  //---------------
  template <typename R, typename I, unsigned M, unsigned N>
  struct axis_pow2_base : public axis_base<R, I, M, N> {
    static constexpr unsigned c_M_end = 1 << M;
    static constexpr unsigned c_N_end = 1 << N;

    axis_pow2_base(R min, R max) : axis_base<R, I, M, N>(min, max, c_M_end, c_N_end) {}

    unsigned size_of_M() const { return c_M_end; }
    unsigned size_of_N() const { return c_N_end; }
  };

  // axis_pow2_u1
  //-------------
  template <typename R, typename I, unsigned M, unsigned N>
  struct axis_pow2_u1 : public axis_pow2_base<R, I, M, N> {
    static constexpr I c_M_mask = (1 << M) - 1;
    static constexpr I c_N_mask = (1 << N) - 1;

    axis_pow2_u1(R min, R max) : axis_pow2_base<R, I, M, N>(min, max) {}

    I R_to_M_bin_safe(R r) const { return this->R_to_M_bin(r) & c_M_mask; }
    I R_to_N_bin_safe(R r) const { return this->R_to_N_bin(r) & c_N_mask; }

    typename axis_base<R, I, M, N>::I_pair Rminmax_to_N_bins(R rmin, R rmax) const {
      return typename axis_base<R, I, M, N>::I_pair(R_to_N_bin_safe(rmin), (this->R_to_N_bin(rmax) + I{1}) & c_N_mask);
    }

    typename axis_base<R, I, M, N>::I_pair Rrdr_to_N_bins(R r, R dr) const { return Rminmax_to_N_bins(r - dr, r + dr); }
    I next_N_bin(I bin) const { return (bin + 1) & c_N_mask; }
  };

  // axis_pow2
  //----------
  template <typename R, typename I, unsigned M, unsigned N>
  struct axis_pow2 : public axis_pow2_base<R, I, M, N> {
    axis_pow2(R min, R max) : axis_pow2_base<R, I, M, N>(min, max) {}
  };

  // axis
  //-----
  template <typename R, typename I, unsigned M = 8 * sizeof(I), unsigned N = 8 * sizeof(I)>
  struct axis : public axis_base<R, I, M, N> {
    const unsigned m_num_M_bins, m_num_N_bins;

    axis(R min, R max, unsigned n_bins)
        : axis_base<R, I, M, N>(min, max, n_bins << this->c_M2N_shift, n_bins),
          m_num_M_bins(n_bins << this->c_M2N_shift),
          m_num_N_bins(n_bins) {}

    axis(R min, R max, R bin_width) {
      R extent = max - min;
      unsigned n_bins = std::ceil(extent / bin_width);
      R extra = (n_bins * bin_width - extent) / 2;

      axis(min - extra, max + extra, n_bins);
    }

    unsigned size_of_M() const { return m_num_M_bins; }
    unsigned size_of_N() const { return m_num_N_bins; }
  };

  // binnor
  //---------------
  // C - bin content type
  // A1, A2 - axis types
  // NB_first, NB_count - number of bits for storage of { first, count } pairs

  template <typename C, typename A1, typename A2, unsigned NB_first = 8 * sizeof(C), unsigned NB_count = 8 * sizeof(C)>
  struct binnor {
    static_assert(std::is_same<typename A1::real_t, typename A2::real_t>());

    static constexpr unsigned c_A2_Mout_mask = ~(((1 << A2::c_M2N_shift) - 1) << A1::c_M);

    // Pair of axis bin indices.
    struct B_pair {
      typename A1::index_t bin1 : A1::c_M;
      typename A2::index_t bin2 : A2::c_M;

      B_pair() : bin1(0), bin2(0) {}
      B_pair(typename A1::index_t i1, typename A2::index_t i2) : bin1(i1), bin2(i2) {}
    };

    // Bin content pair.
    struct C_pair {
      C first : NB_first;
      C count : NB_count;

      C_pair() : first(0), count(0) {}
      C_pair(C f, C c) : first(f), count(c) {}

      C end() const { return first + count; }
    };

    const A1 &m_a1;
    const A2 &m_a2;
    std::vector<B_pair> m_cons;
    std::vector<C_pair> m_bins;
    std::vector<C> m_ranks;

    binnor(const A1 &a1, const A2 &a2) : m_a1(a1), m_a2(a2), m_bins(m_a1.size_of_N() * m_a2.size_of_N()) {}

    // Access

    B_pair m_bin_to_n_bin(B_pair m_bin) { return {m_a1.M_bin_to_N_bin(m_bin.bin1), m_a2.M_bin_to_N_bin(m_bin.bin2)}; }

    B_pair get_n_bin(typename A1::index_t n1, typename A2::index_t n2) const { return {n1, n2}; }

    B_pair get_n_bin(typename A1::real_t r1, typename A2::real_t r2) const {
      return {m_a1.R_to_N_bin(r1), m_a2.R_to_N_bin(r2)};
    }

    C_pair &ref_content(B_pair n_bin) { return m_bins[n_bin.bin2 * m_a1.size_of_N() + n_bin.bin1]; }

    C_pair get_content(B_pair n_bin) const { return m_bins[n_bin.bin2 * m_a1.size_of_N() + n_bin.bin1]; }

    C_pair get_content(typename A1::index_t n1, typename A2::index_t n2) const {
      return m_bins[n2 * m_a1.size_of_N() + n1];
    }

    C_pair get_content(typename A1::real_t r1, typename A2::real_t r2) const {
      return get_content(m_a1.R_to_N_bin(r1), m_a2.R_to_N_bin(r2));
    }

    // Filling

    void reset_contents() {
      m_bins.assign(m_bins.size(), C_pair());
      m_ranks.clear();
      m_ranks.shrink_to_fit();
    }

    void begin_registration(C n_items) { m_cons.reserve(n_items); }

    void register_entry(typename A1::real_t r1, typename A2::real_t r2) {
      m_cons.push_back({m_a1.R_to_M_bin(r1), m_a2.R_to_M_bin(r2)});
    }

    void register_entry_safe(typename A1::real_t r1, typename A2::real_t r2) {
      m_cons.push_back({m_a1.R_to_M_bin_safe(r1), m_a2.R_to_M_bin_safe(r2)});
    }

    // Do M-binning outside, potentially using R_to_M_bin_safe().
    void register_m_bins(typename A1::index_t m1, typename A2::index_t m2) { m_cons.push_back({m1, m2}); }

    void finalize_registration() {
      // call internal sort, bin building from icc where template instantiation has to be made.

      m_ranks.resize(m_cons.size());
      std::iota(m_ranks.begin(), m_ranks.end(), 0);

      std::sort(m_ranks.begin(), m_ranks.end(), [&](auto &a, auto &b) {
        return (m_cons[a].raw & c_A2_Mout_mask) < (m_cons[b].raw & c_A2_Mout_mask);
      });

      for (C i = 0; i < m_ranks.size(); ++i) {
        C j = m_ranks[i];
        C_pair &c_bin = ref_content(m_bin_to_n_bin(m_cons[j]));
        if (c_bin.count == 0)
          c_bin.first = i;
        ++c_bin.count;

#ifdef DEBUG
        B_pair n_pair = m_bin_to_n_bin(m_cons[j]);
        printf("i=%4u j=%4u  %u %u %u %u\n", i, j, n_pair.bin1, n_pair.bin2, c_bin.first, c_bin.count);
#endif
      }

      // Those could be kept to do preselection when determining search ranges.
      // Especially since additional precision on Axis2 is screened out during sorting.
      m_cons.clear();
      m_cons.shrink_to_fit();
    }
  };

}  // namespace mkfit

#endif

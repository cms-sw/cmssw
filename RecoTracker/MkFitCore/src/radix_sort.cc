#include "RecoTracker/MkFitCore/interface/radix_sort.h"

#include <array>
#include <numeric>

namespace mkfit {

  // --- Driver function

  template <typename V, typename R>
  void radix_sort<V, R>::sort(const std::vector<V>& values, std::vector<R>& ranks) {
    if (values.empty()) {
      ranks.clear();
      return;
    }

    rank_t histos[c_NBytes * 256] = {0};

    histo_loop(values, histos);
    radix_loop(values, histos, ranks);
  }

  // --- Histogramming

  template <typename V, typename R>
  void radix_sort<V, R>::histo_loop(const std::vector<V>& values, rank_t* histos) {
    // Create histograms (counters). Counters for all passes are created in one run.
    ubyte_t* p = (ubyte_t*)values.data();
    ubyte_t* pe = p + (values.size() * c_NBytes);
    std::array<rank_t*, c_NBytes> ha;
    for (rank_t j = 0; j < c_NBytes; ++j)
      ha[j] = &histos[j << 8];
    while (p != pe) {
      for (rank_t j = 0; j < c_NBytes; ++j)
        ha[j][*p++]++;
    }
  }

  // --- Radix

  template <typename V, typename R>
  void radix_sort<V, R>::radix_loop(const std::vector<V>& values, rank_t* histos, std::vector<R>& ranks) {
    const rank_t nb = values.size();
    rank_t* link[256];
    ranks.resize(nb);
    std::vector<rank_t> ranks2(nb);
    bool ranks_are_invalid = true;
    // Radix sort, j is the pass number (0=LSB, 3=MSB)
    for (rank_t j = 0; j < c_NBytes; j++) {
      // Shortcut to current counters
      rank_t* cur_count = &histos[j << 8];

      // Get first byte - if that byte's counter equals nb, all values are the same.
      ubyte_t first_entry_val = *(((ubyte_t*)values.data()) + j);
      if (cur_count[first_entry_val] != nb) {
        // Create offsets
        link[0] = ranks2.data();
        for (rank_t i = 1; i < 256; i++)
          link[i] = link[i - 1] + cur_count[i - 1];

        // Perform Radix Sort
        ubyte_t* input_bytes = (ubyte_t*)values.data();
        input_bytes += j;
        if (ranks_are_invalid) {
          for (rank_t i = 0; i < nb; i++)
            *link[input_bytes[i << 2]]++ = i;
          ranks_are_invalid = false;
        } else {
          rank_t* indices = ranks.data();
          rank_t* indices_end = indices + nb;
          while (indices != indices_end) {
            rank_t id = *indices++;
            *link[input_bytes[id << 2]]++ = id;
          }
        }

        // Swap ranks - valid indices are in ranks after the swap.
        ranks.swap(ranks2);
      }
    }
    // If all values are equal, fill ranks with sequential integers.
    if (ranks_are_invalid)
      std::iota(ranks.begin(), ranks.end(), 0);
  }

  // Instantiate supported sort types.
  template class radix_sort<unsigned int, unsigned int>;
  template class radix_sort<unsigned int, unsigned short>;
}  // namespace mkfit

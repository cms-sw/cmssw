#ifndef L1Trigger_L1TMuonEndCapPhase2_DataUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_DataUtils_h

#include <array>
#include <vector>
#include <type_traits>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::data {

  // Merge-Sort
  template <typename T, typename C>
  void swapWires(T arr[], const unsigned int& wire_1, const unsigned int& wire_2, const C& comparator) {
    int result = comparator(arr[wire_1], arr[wire_2]);

    if (result == 1) {
      auto temp = arr[wire_1];
      arr[wire_1] = arr[wire_2];
      arr[wire_2] = temp;
    }
  }

  template <typename T, typename C>
  void mergesortBlock(T arr[],
                      const unsigned int& offset,
                      const unsigned int& step,
                      const unsigned int& block_begin,
                      const unsigned int& block_end,
                      const unsigned int& first_n,
                      const C& comparator) {
    auto wire_offset = offset + block_begin;
    auto wire_cutoff = first_n + block_begin;
    auto wire_1 = wire_offset;
    auto wire_2 = wire_1 + step;

    // Loop pairs
    while (wire_2 < block_end) {
      // Trim results
      if (first_n > 0 && wire_cutoff < block_end) {
        bool wire_needed = (wire_offset <= wire_1) && (wire_1 <= wire_cutoff);

        if (!wire_needed) {
          break;
        }
      }

      // Swap Wires
      swapWires(arr, wire_1, wire_2, comparator);

      // Calculate next wire_1
      if (step == 1) {
        wire_1 = wire_2 + 1;
      } else {
        wire_1 = wire_1 + 1;
      }

      // Calculate next wire_2
      wire_2 = wire_1 + step;
    }
  }

  template <typename T, typename C>
  void mergesort(T arr[], const unsigned int& arr_size, const unsigned int& first_n, const C& comparator) {
    // Sort
    auto n_pairs = static_cast<unsigned int>(arr_size / 2);

    for (unsigned int i = 0; i < n_pairs; ++i) {
      swapWires(arr, 2 * i, 2 * i + 1, comparator);
    }

    // Merge
    auto offset = 0u;
    auto step = 2u;
    auto block_size = step * 2;

    // Loop block sizes
    while (true) {
      // Loop step sizes
      // If the offset is greater than the amount of wires to keep
      // there's no need to continue, since (offset)-wires are known
      // to not contribute to the end result
      while (true) {
        // Loop blocks
        auto block_begin = 0u;
        auto block_end = block_size;

        while (block_begin < arr_size) {
          // Constrain block_end
          if (block_end > arr_size)
            block_end = arr_size;

          // Merge block
          mergesortBlock(arr, offset, step, block_begin, block_end, first_n, comparator);

          // Move to next block
          block_begin = block_end;
          block_end = block_end + block_size;
        }

        // Decrease step
        if (step > 2) {
          // For each pass we are certain of the local min and max
          // so skip 2 wires and reduce the step
          offset = offset + 2;
          step = step - 2;
        } else if (step == 2) {
          // For final pass we are certain of the global min and max
          // so skip 1 wire and compare wires 1 to 1, the last value
          // will be left without a partner; naturally since
          // it's the global min
          offset = 1;
          step = 1;
        } else {
          // Short-Circuit: Done
          break;
        }
      }

      // Short-Circuit: No more wires
      if (block_size >= arr_size)
        break;

      // Double the block size
      offset = 0;
      step = block_size;
      block_size = step * 2;
    }
  }

  template <typename T, typename C>
  void mergesort(T arr[], const unsigned int& arr_size, const C& comparator) {
    mergesort(arr, arr_size, 0, comparator);
  }

  // Median Calculation
  template <typename T>
  T getMedianOfSorted(T arr[], const unsigned int& arr_size) {
    T mid;

    if ((arr_size % 2) == 0) {
      const auto& top = arr[arr_size / 2];
      const auto& bot = arr[arr_size / 2 - 1];
      mid = (top + bot) >> 1;  // Mid = (Top + Bot) / 2
    } else {
      mid = arr[(arr_size - 1) / 2];
    }

    return mid;
  }
}  // namespace emtf::phase2::data

#endif  // L1Trigger_L1TMuonEndCapPhase2_DataUtils_h not defined

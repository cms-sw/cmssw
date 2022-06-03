#ifndef BITONIC_HYBRID_REF_H
#define BITONIC_HYBRID_REF_H

#include <algorithm>
#include <cassert>

namespace hybridBitonicSortUtils {
  inline unsigned int PowerOf2LessThan(unsigned int n) {
    unsigned int i = 1;
    unsigned int prev = 1;
    if (n <= 1)
      return n;
    while (i < n) {
      i <<= 1;
      if (i < n) {
        prev = i;
      } else {
        return prev;
      }
    }
    // shouldn't happen
    assert(false);
  }

  template <typename T>
  void compAndSwap(T a[], int i, int j, bool dir) {
    if (dir) {
      if (a[j] < a[i])
        std::swap(a[i], a[j]);
    } else {
      if (a[i] < a[j])
        std::swap(a[i], a[j]);
    }
  }

  inline unsigned int bitonicMergeLatencyRef(unsigned int nIn) {
    if (nIn <= 1)
      return 0;
    return 1 +
           std::max(bitonicMergeLatencyRef(PowerOf2LessThan(nIn)), bitonicMergeLatencyRef(nIn - PowerOf2LessThan(nIn)));
  }

  inline unsigned int bitonicSortLatencyRef(unsigned int nIn, unsigned int nOut) {
    if (nIn <= 1)
      return 0;
    unsigned int sort1Size = nIn / 2, sort2Size = nIn - sort1Size;
    unsigned int sort1Latency = bitonicSortLatencyRef(sort1Size, nOut);
    unsigned int sort2Latency = bitonicSortLatencyRef(sort2Size, nOut);
    unsigned int mergeLatency = bitonicMergeLatencyRef(std::min(sort1Size, nOut) + std::min(sort2Size, nOut));
    return std::max(sort1Latency, sort2Latency) + mergeLatency;
  }

  inline unsigned int hybridBitonicSortLatencyRef(unsigned int nIn, unsigned int nOut) {
    if (nIn <= 1)
      return 0;
    if (nIn == 5 || nIn == 6)
      return 3;
    if (nIn == 12)
      return 8;
    if (nIn == 13)
      return 9;
    unsigned int sort1Size = nIn / 2, sort2Size = nIn - sort1Size;
    unsigned int sort1Latency = hybridBitonicSortLatencyRef(sort1Size, nOut);
    unsigned int sort2Latency = hybridBitonicSortLatencyRef(sort2Size, nOut);
    unsigned int mergeLatency = bitonicMergeLatencyRef(std::min(sort1Size, nOut) + std::min(sort2Size, nOut));
    return std::max(sort1Latency, sort2Latency) + mergeLatency;
  }

  // may be specialized for different types if needed
  template <typename T>
  void clear(T& t) {
    t.clear();
  }

}  // namespace hybridBitonicSortUtils

template <typename T>
void hybridBitonicMergeRef(T a[], int N, int low, bool dir) {
  int k = hybridBitonicSortUtils::PowerOf2LessThan(N);
  int k2 = N - k;
  if (N > 1) {
    for (int i = low; i < low + k; i++) {
      if (i + k < low + N)
        hybridBitonicSortUtils::compAndSwap(a, i, i + k, dir);
    }
    if (N > 2) {
      hybridBitonicMergeRef(a, k, low, dir);
      hybridBitonicMergeRef(a, k2, low + k, dir);
    }
  }
}

template <typename T>
void check_sorted(T a[], int N, int low, bool dir) {
  bool ok = true;
  if (dir) {
    for (int i = 1; i < N; ++i)
      ok = ok && (!(a[low + i - 1] > a[low + i]));
  } else {
    for (int i = 1; i < N; ++i)
      ok = ok && (!(a[low + i - 1] < a[low + i]));
  }
  if (!ok) {
    printf("ERROR in sorting[N=%d,low=%d,dir=%d]: ", N, low, int(dir));
    //for (int i = 0; i < N; ++i) printf("%d[%s]  ", a[low+i].intPt(), a[low+i].pack().to_string(16).c_str());
    //for (int i = 0; i < N; ++i) printf("%d  ", a[low+i]);
    printf("\n");
    fflush(stdout);
    assert(ok);
  }
}

template <typename T>
void hybridBitonicSortRef(T a[], int N, int low, bool dir, bool hybrid) {
  if (hybrid) {  // sorting networks defined by hand for a few cases
    switch (N) {
      case 2:
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        return;
      case 3:
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        //check_sorted(a, N, low, dir);
        return;
      case 4:
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 3, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 2, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 3, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        //check_sorted(a, N, low, dir);
        return;
      case 5:
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 3, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 3, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 4, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 2, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 4, dir);
        //--
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 4, dir);
        //--
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 3, dir);
        //check_sorted(a, N, low, dir);
        return;
      case 6:
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 3, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 3, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 4, low + 5, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 3, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 5, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 1, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 5, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 4, dir);
        //check_sorted(a, N, low, dir);
        return;
      case 12:
        for (int i = 0; i < 12; i += 2) {
          hybridBitonicSortUtils::compAndSwap(a, low + i, low + i + 1, dir);
        }
        //---
        for (int i = 0; i < 12; i += 4) {
          hybridBitonicSortUtils::compAndSwap(a, low + i + 0, low + i + 2, dir);
          hybridBitonicSortUtils::compAndSwap(a, low + i + 1, low + i + 3, dir);
        }
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 5, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 11, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 9, low + 10, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 6, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 9, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 4, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 7, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 5, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 9, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 11, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 8, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 3, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 8, low + 9, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 5, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 6, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 9, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 8, dir);
        //check_sorted(a, N, low, dir);
        return;
      case 13:
        for (int i = 0; i + 1 < 13; i += 2) {
          hybridBitonicSortUtils::compAndSwap(a, low + i, low + i + 1, dir);
        }
        //---
        for (int i = 0; i + 3 < 13; i += 4) {
          hybridBitonicSortUtils::compAndSwap(a, low + i + 0, low + i + 2, dir);
          hybridBitonicSortUtils::compAndSwap(a, low + i + 1, low + i + 3, dir);
        }
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 5, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 7, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 8, low + 12, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 0, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 9, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 11, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 4, low + 12, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 2, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 12, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 11, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 4, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 6, low + 9, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 1, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 6, low + 12, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 9, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 2, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 5, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 6, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 9, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 10, low + 12, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 9, low + 12, dir);
        //---
        hybridBitonicSortUtils::compAndSwap(a, low + 3, low + 4, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 5, low + 6, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 7, low + 8, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 9, low + 10, dir);
        hybridBitonicSortUtils::compAndSwap(a, low + 11, low + 12, dir);
        //check_sorted(a, N, low, dir);
        return;
    }
  }

  // general case
  if (N > 1) {
    int lowerSize = N / 2;
    int upperSize = N - N / 2;
    bool notDir = not dir;
    hybridBitonicSortRef(a, lowerSize, low, notDir, hybrid);
    hybridBitonicSortRef(a, upperSize, low + lowerSize, dir, hybrid);
    hybridBitonicMergeRef(a, N, low, dir);
    //check_sorted(a, N, low, dir);
  }
}

template <typename T>
void hybrid_bitonic_sort_and_crop_ref(
    unsigned int nIn, unsigned int nOut, const T in[], T out[], bool hybrid = true) {  // just an interface
  T work[nIn];
  for (unsigned int i = 0; i < nIn; ++i) {
    work[i] = in[i];
  }
  hybridBitonicSortRef(work, nIn, 0, false, hybrid);
  for (unsigned int i = 0; i < nOut; ++i) {
    out[i] = work[i];
  }
}

template <typename T>
void folded_hybrid_bitonic_sort_and_crop_ref(
    unsigned int nIn, unsigned int nOut, const T in[], T out[], bool hybrid = true) {  // just an interface
  unsigned int nHalf = (nIn + 1) / 2;
  T work[nHalf], halfsorted[nHalf];
  //printf("hybrid sort input %u items: ", nIn);
  //for (int i = 0; i < nIn; ++i) printf("%d.%03d  ", work[i].intPt(), work[i].intEta());
  //for (int i = 0; i < nIn; ++i) if (in[i].hwPt) printf("[%d]%s  ", i, in[i].pack().to_string(16).c_str());
  //printf("\n");
  //fflush(stdout);
  for (int o = 1; o >= 0; --o) {
    for (unsigned int i = 0; i < nHalf && 2 * i + o < nIn; ++i) {
      work[i] = in[2 * i + o];
    }
    if ((nIn % 2 == 1) && (o == 1)) {
      hybridBitonicSortUtils::clear(work[nHalf - 1]);
    }
    hybridBitonicSortRef(work, nHalf, 0, false, hybrid);
    //printf("hybrid sort offset %d with %u items: ", o, nHalf);
    //for (int i = 0; i < nHalf; ++i) printf("%d.%03d  ", work[i].intPt(), work[i].intEta());
    //for (int i = 0; i < nHalf; ++i) printf("%s  ", work[i].pack().to_string(16).c_str());
    //printf("\n");
    //fflush(stdout);
    for (unsigned int i = 1; i < nHalf; ++i)
      assert(!(work[i - 1] < work[i]));
    if (o == 1) {
      for (unsigned int i = 0; i < nHalf; ++i) {
        halfsorted[i] = work[i];
      }
    }
  }
  // now merge work with the reversed of half-sorted
  unsigned int nMerge = std::min(nOut, nHalf);
  T tomerge[2 * nMerge];
  for (unsigned int i = 0; i < nMerge; ++i) {
    tomerge[nMerge - i - 1] = halfsorted[i];
    tomerge[nMerge + i] = work[i];
  }
  //printf("hybrid sort tomerge %u items before: ", 2*nMerge);
  //for (int i = 0; i < 2*nMerge; ++i) printf("%d.%03d  ", tomerge[i].intPt(), tomerge[i].intEta());
  //for (int i = 0; i < 2*nMerge; ++i) printf("%s  ", tomerge[i].pack().to_string(16).c_str());
  //printf("\n");
  hybridBitonicMergeRef(tomerge, 2 * nMerge, 0, false);
  //printf("hybrid sort tomerge %u items after:  ", 2*nMerge);
  //for (int i = 0; i < 2*nMerge; ++i) printf("%d.%03d  ", tomerge[i].intPt(), tomerge[i].intEta());
  //for (int i = 0; i < nOut; ++i) printf("%s  ", tomerge[i].pack().to_string(16).c_str());
  //printf("\n");
  //fflush(stdout);
  for (unsigned int i = 0; i < nOut; ++i) {
    out[i] = tomerge[i];
    if (i > 0)
      assert(!(out[i - 1] < out[i]));
  }
}

#endif

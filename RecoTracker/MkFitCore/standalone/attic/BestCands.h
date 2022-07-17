#ifndef RecoTracker_MkFitCore_standalone_attic_BestCands_h
#define RecoTracker_MkFitCore_standalone_attic_BestCands_h

#include "Config.h"

#include <cstdio>
#include <limits>

namespace CandsGPU {

  constexpr int trkIdx_sentinel = -1;
  constexpr int hitIdx_sentinel = -1;
  constexpr int nhits_sentinel = -1;
  constexpr float chi2_sentinel = std::numeric_limits<float>::max();

  template <typename T>
  __device__ void swap_values(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
  }

  template <int MaxCandsPerSeed, int BlockSize>
  struct BestCands {
    // AoS would generate bank conflicts when used in SM
    int trkIdx[MaxCandsPerSeed][BlockSize];
    int hitIdx[MaxCandsPerSeed][BlockSize];
    int nhits[MaxCandsPerSeed][BlockSize];
    float chi2[MaxCandsPerSeed][BlockSize];

    __device__ void reset(int itrack);
    __device__ void update(int itrack, int cand_trIdx, int cand_hitIdx, int cand_nhits, float cand_chi2);
    __device__ void heapify(int itrack, int idx, int heap_size);
    __device__ int left(int idx);
    __device__ int right(int idx);

    __device__ bool better(int icand_fst, int fst, int icand_snd, int snd);

    __device__ void heap_sort(int icand, int heap_size);
    __device__ void merge_cands_for_seed(int iseed, int icand);
    __device__ void swap_nodes(int icand_fst, int fst, int icand_snd, int snd);
    __device__ void copy_node(int icand_fst, int fst, int icand_snd, int snd);

    __device__ int count_valid_cands(int itrack);

    // TODO: Should really return a IdxChi2List
    __device__ void get_cand_info(
        const int tid, const int cid, int& my_trkIdx, int& my_hitIdx, int& my_nhits, float& my_chi2);
    __device__ int get_nhits(const int tid, const int cid) { return nhits[cid][tid]; }

    __device__ void print_heap(const int tid);
  };

  template <int M, int B>
  __device__ void BestCands<M, B>::reset(int itrack) {
    for (auto j = 0; j < M; ++j) {
      trkIdx[j][itrack] = trkIdx_sentinel;
      hitIdx[j][itrack] = hitIdx_sentinel;
      nhits[j][itrack] = nhits_sentinel;
      chi2[j][itrack] = chi2_sentinel;
    }
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::update(int itrack, int cand_trIdx, int cand_hitIdx, int cand_nhits, float cand_chi2) {
    if (cand_nhits < nhits[0][itrack])
      return;
    if (cand_chi2 > chi2[0][itrack])
      return;
    trkIdx[0][itrack] = cand_trIdx;
    hitIdx[0][itrack] = cand_hitIdx;
    nhits[0][itrack] = cand_nhits;
    chi2[0][itrack] = cand_chi2;

    heapify(itrack, 0, M);
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::print_heap(const int tid) {
    for (int cid = 0; cid < M; cid++) {
      printf(">>>>> tid %d rowIdx %d hitIdx %d nhits %d chi2 %f\n",
             tid,
             cid,
             hitIdx[cid][tid],
             nhits[cid][tid],
             chi2[cid][tid]);
    }
  }

  template <int M, int B>
  __device__ int BestCands<M, B>::left(int idx) {
    return (++idx << 1) - 1;
  }

  template <int M, int B>
  __device__ int BestCands<M, B>::right(int idx) {
    return ++idx << 1;
  }

  template <int M, int B>
  __device__ bool BestCands<M, B>::better(int icand_fst, int fst, int icand_snd, int snd) {
    return (nhits[fst][icand_fst] > nhits[snd][icand_snd]) ||
           ((nhits[fst][icand_fst] == nhits[snd][icand_snd]) && (chi2[fst][icand_fst] < chi2[snd][icand_snd]));
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::swap_nodes(int icand_fst, int fst, int icand_snd, int snd) {
    swap_values(trkIdx[fst][icand_fst], trkIdx[snd][icand_snd]);
    swap_values(hitIdx[fst][icand_fst], hitIdx[snd][icand_snd]);
    swap_values(nhits[fst][icand_fst], nhits[snd][icand_snd]);
    swap_values(chi2[fst][icand_fst], chi2[snd][icand_snd]);
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::copy_node(int icand_fst, int fst, int icand_snd, int snd) {
    trkIdx[snd][icand_snd] = trkIdx[fst][icand_fst];
    hitIdx[snd][icand_snd] = hitIdx[fst][icand_fst];
    nhits[snd][icand_snd] = nhits[fst][icand_fst];
    chi2[snd][icand_snd] = chi2[fst][icand_fst];
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::heapify(int icand, int idx, int heap_size) {
    // We want to move idx down so the smallest value is at the root
    int smallest = -1;
    while (idx != smallest) {
      if (idx < 0 || idx >= heap_size / 2)
        return;

      smallest = idx;
      if (heap_size > left(idx) && better(icand, smallest, icand, left(idx)))
        smallest = left(idx);
      if (heap_size > right(idx) && better(icand, smallest, icand, right(idx)))
        smallest = right(idx);

      if (smallest != idx) {
        swap_nodes(icand, smallest, icand, idx);
        idx = smallest;
        smallest = -1;
      }
    }
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::merge_cands_for_seed(int iseed, int icand) {
    int itrack = iseed * M + icand;
// TODO: Need a better way to reduce candidates.
//       So far, binary tree reduction is a bit slower than the naive approach
#if 1
    if (icand) {
      heap_sort(itrack, M);
    }
    __syncthreads();  // cand 0 waits;
    if (icand)
      return;  // reduction by the first cand of each seed

    for (int i = itrack + 1; i < itrack + M; ++i) {  // over cands
      for (int j = 0; j < M; ++j) {                  // inside heap
        if (better(i, j, itrack, 0)) {
          copy_node(i, j, itrack, 0);
          heapify(itrack, 0, M);
        } else {
          break;
        }
      }
    }
    heap_sort(itrack, M);
    __syncthreads();  // TODO: Volta: sync only on MaxCandsPerSeeds threads
#else

    for (int step = 2; step <= Config::maxCandsPerSeed; step <<= 1) {
      if (icand % step == step / 2) {
        heap_sort(itrack, M);
      }
      __syncthreads();

      if (icand % step == 0) {
        int i = itrack + step / 2;
        if ((i < iseed * M + M) && (i < B) && (icand + step / 2 < M)) {
          for (int j = 0; j < M; ++j) {  // inside heap
            if (better(i, j, itrack, 0)) {
              copy_node(i, j, itrack, 0);
              heapify(itrack, 0, M);
            } else {
              break;
            }
          }
        }
      }
      //__syncthreads();
    }

    if (icand == 0) {
      heap_sort(itrack, M);
    }
    __syncthreads();
#endif
  }

  template <int M, int B>
  __device__ void BestCands<M, B>::heap_sort(int icand, int heap_size) {
    int num_unsorted_elts = heap_size;
    // Assume that we have a heap with the worst one at the root.
    for (int i = heap_size - 1; i > 0; --i) {
      swap_nodes(icand, 0, icand, i);  // worst at the end
      heapify(icand, 0, --num_unsorted_elts);
    }
  }

  template <int MaxCandsPerSeed, int BlockSize>
  __device__ void BestCands<MaxCandsPerSeed, BlockSize>::get_cand_info(
      const int tid, const int cid, int& my_trkIdx, int& my_hitIdx, int& my_nhits, float& my_chi2) {
    if (cid < MaxCandsPerSeed && tid < BlockSize) {
      my_trkIdx = trkIdx[cid][tid];
      my_hitIdx = hitIdx[cid][tid];
      my_nhits = nhits[cid][tid];
      my_chi2 = chi2[cid][tid];
    }
  }

  template <int M, int B>
  __device__ int BestCands<M, B>::count_valid_cands(int itrack) {
    int count = 0;
    for (int i = 0; i < M; ++i) {
      if (trkIdx[i][itrack] != trkIdx_sentinel)
        ++count;
    }
    return count;
  }

}  // namespace CandsGPU

#endif  // _BEST_CANDS_H_

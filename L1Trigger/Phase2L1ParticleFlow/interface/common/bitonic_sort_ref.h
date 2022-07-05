#ifndef BITONIC_NEW_H
#define BITONIC_NEW_H

#include <algorithm>
#include <cassert>

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

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
void bitonicMerge(T in[], int InSize, T out[], int OutSize, bool dir) {
  //printDebug;

  // size == 1 -> pass through
  if (InSize <= 1) {
    for (int i = 0; i < std::min(InSize, OutSize); ++i)
      out[i] = in[i];
    return;
  }

  if (InSize > 1) {
    int LowerSize = PowerOf2LessThan(InSize);  //-- LowerSize >= Size / 2
    int UpperSize = InSize - LowerSize;        //-- UpperSize < LowerSiz

    assert(UpperSize >= 0);
    assert(UpperSize <= LowerSize);

    for (int i = 0; i < UpperSize; ++i) {
      if ((in[i] > in[i + LowerSize]) == dir) {
        // this checks should refer to the comments, "just needs to be long enough"
        if (i < OutSize)
          out[i] = in[i + LowerSize];
        if (i + LowerSize < OutSize)
          out[i + LowerSize] = in[i];
      } else {
        if (i < OutSize)
          out[i] = in[i];
        if (i + LowerSize < OutSize)
          out[i + LowerSize] = in[i + LowerSize];
      }
    }

    // Copy the residual at the end. This limits the sorting in the overall descending direction (if out != in).
    if (LowerSize > UpperSize) {
      for (int i = UpperSize; i < LowerSize; ++i) {
        if (i < OutSize)
          out[i] = in[i];
      }
    }

    T out2[LowerSize];
    bitonicMerge(out, LowerSize, out2, LowerSize, dir);

    T out3[UpperSize];
    bitonicMerge(out + LowerSize, UpperSize, out3, UpperSize, dir);

    // copy back to out; direction dependent.
    if (dir)  // ascending -- Copy up to OutSize
    {
      for (int i = 0; i < OutSize; ++i) {
        if (i < UpperSize)
          out[OutSize - i - 1] = out3[UpperSize - i - 1];
        else
          out[OutSize - i - 1] = out2[LowerSize - i - 1 + UpperSize];
      }

    } else {  //descending
      for (int i = 0; i < LowerSize; ++i) {
        if (i < OutSize)
          out[i] = out2[i];
      }
      for (int i = LowerSize; i < OutSize; ++i)
        out[i] = out3[i - LowerSize];
    }

  }  // InSize>1

}  // bitonicMerge

template <typename T>
void bitonicSort(const T in[], int Start, int InSize, T out[], int OutSize, bool dir) {
  assert(OutSize > 0);
  if (InSize <= 1)  // copy in-> out and exit
  {
    for (int i = 0; i < std::min(InSize, OutSize); ++i)
      out[i] = in[i + Start];
    return;
  }

  int LowerInSize = InSize / 2;
  int UpperInSize = InSize - LowerInSize;  //-- UpperSize >= LowerSize

  int LowerOutSize = std::min(OutSize, LowerInSize);
  int UpperOutSize = std::min(OutSize, UpperInSize);

  // sorted output
  T OutTmp[LowerOutSize + UpperOutSize];

  // sort first half
  bitonicSort(in,
              Start,
              LowerInSize,
              OutTmp,
              LowerOutSize,
              not dir);  // the not dir enforce the sorting in overall descending direction.

  // sort second half
  bitonicSort(in, Start + LowerInSize, UpperInSize, OutTmp + LowerOutSize, UpperOutSize, dir);

  // create a temporary output vector "large enough" and then copy back
  int OutSize2 = LowerOutSize + UpperOutSize;
  T outTmp2[OutSize2];
  bitonicMerge(OutTmp, LowerOutSize + UpperOutSize, outTmp2, OutSize2, dir);
  //copy back to out the first OutSize
  for (int i = 0; i < OutSize; ++i) {
    if (dir) {  //ascending
      out[OutSize - 1 - i] = outTmp2[OutSize2 - 1 - i];
    } else {  //descending
      out[i] = outTmp2[i];
    }
  }
}

template <typename T>
void bitonic_sort_and_crop_ref(unsigned int nIn, unsigned int nOut, const T in[], T out[]) {  // just an interface
  bitonicSort(in, 0, nIn, out, nOut, 0);
}
#endif

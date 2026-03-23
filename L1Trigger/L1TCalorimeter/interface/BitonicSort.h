#ifndef BitonicSort_h
#define BitonicSort_h

#include <cstdint>
#include <vector>

enum sort_direction { up, down };

// DECLARE!
template <typename T, typename Greater, typename Itr>
void BitonicSort(sort_direction aDir, Itr aDataStart, Itr aDataEnd);
template <typename T, typename Greater, typename Itr>
void BitonicMerge(sort_direction aDir, Itr aDataStart, Itr aDataEnd);
//DEFINE!

//SORT
template <typename T, typename Greater, typename Itr>
void BitonicSort(sort_direction aDir, Itr aDataStart, Itr aDataEnd) {
  uint32_t lSize(aDataEnd - aDataStart);
  if (lSize > 1) {
    Itr lMidpoint(aDataStart + (lSize >> 1));
    if (aDir == down) {
      BitonicSort<T, Greater>(up, aDataStart, lMidpoint);
      BitonicSort<T, Greater>(down, lMidpoint, aDataEnd);
    } else {
      BitonicSort<T, Greater>(down, aDataStart, lMidpoint);
      BitonicSort<T, Greater>(up, lMidpoint, aDataEnd);
    }
    BitonicMerge<T, Greater>(aDir, aDataStart, aDataEnd);
  }
}

//MERGE
template <typename T, typename Greater, typename Itr>
void BitonicMerge(sort_direction aDir, Itr aDataStart, Itr aDataEnd) {
  uint32_t lSize(aDataEnd - aDataStart);
  if (lSize > 1) {
    uint32_t lPower2(1);
    while (lPower2 < lSize)
      lPower2 <<= 1;

    Itr lMidpoint(aDataStart + (lPower2 >> 1));
    Itr lFirst(aDataStart);
    Itr lSecond(lMidpoint);

    const Greater g;
    for (; lSecond != aDataEnd; ++lFirst, ++lSecond) {
      if (g((*lFirst), (*lSecond)) == (aDir == up)) {
        std::swap(*lFirst, *lSecond);
      }
    }

    BitonicMerge<T, Greater>(aDir, aDataStart, lMidpoint);
    BitonicMerge<T, Greater>(aDir, lMidpoint, aDataEnd);
  }
}

#endif

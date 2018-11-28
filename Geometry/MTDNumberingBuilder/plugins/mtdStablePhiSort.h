#ifndef MTDStablePhiSort_H
#define MTDStablePhiSort_H

#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"

template<class RandomAccessIterator, class Extractor>
void mtdStablePhiSort(RandomAccessIterator begin, RandomAccessIterator end, const Extractor& extr)
{
    trackerStablePhiSort(begin, end, extr);
}


#endif

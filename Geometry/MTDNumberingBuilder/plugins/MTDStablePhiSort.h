#ifndef MTDStablePhiSort_H
#define MTDStablePhiSort_H

#include "Geometry/TrackerNumberingBuilder/plugins/TrackerStablePhiSort.h"

template<class RandomAccessIterator, class Extractor>
void mtdStablePhiSort(RandomAccessIterator begin, RandomAccessIterator end, const Extractor& extr)
{
    trackerStablePhiSort(begin, end, extr);
}


#endif

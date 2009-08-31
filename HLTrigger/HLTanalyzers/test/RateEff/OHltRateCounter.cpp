#include "OHltRateCounter.h"

OHltRateCounter::OHltRateCounter(unsigned int size) {
  vector<int> itmp;
  for (unsigned int i=0;i<size;i++) {
    iCount.push_back(0);
    sPureCount.push_back(0);
    pureCount.push_back(0);
    prescaleCount.push_back(0);

    itmp.push_back(0);
  }
  for (unsigned int j=0;j<size;j++)
    overlapCount.push_back(itmp);
}

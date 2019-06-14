#ifndef _ClusterData_h_
#define _ClusterData_h_

#include "FWCore/Utilities/interface/VecArray.h"
#include <utility>

class ClusterData {
public:
  using ArrayType = edm::VecArray<std::pair<int, int>, 9>;
  ArrayType size;
  bool isStraight, isComplete, hasBigPixelsOnlyInside;
};

#endif

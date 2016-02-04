#ifndef _ClusterData_h_
#define _ClusterData_h_

#include <utility>
#include <vector>

class ClusterData
{
 public:
   bool isStraight,isComplete, hasBigPixelsOnlyInside; 
   std::vector<std::pair<int,int> > size;
};

#endif

#ifndef _ClusterData_h_
#define _ClusterData_h_

#include <utility>

class ClusterData
{
 public:
   bool isStraight,isComplete; 
   std::pair<unsigned short int,short int> size;
};

#endif

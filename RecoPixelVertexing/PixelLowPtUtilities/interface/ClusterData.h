#ifndef _ClusterData_h_
#define _ClusterData_h_

#include <utility>

class ClusterData
{
 public:
   bool isInBarrel, isNormalOriented, isStraight,isComplete, isUnlocked;
   bool isXBorder;
   int posBorder;
   std::pair<float,float> tangent;
   std::pair<unsigned short int,short int> size;
};

#endif

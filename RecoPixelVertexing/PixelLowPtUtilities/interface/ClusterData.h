#ifndef _ClusterData_h_
#define _ClusterData_h_

#include <utility>
using namespace std;

class ClusterData
{
 public:
   bool isInBarrel, isNormalOriented, isStraight,isComplete, isUnlocked;
   bool isXBorder;
   int posBorder;
   pair<float,float> tangent;
   pair<unsigned short int,short int> size;
};

#endif

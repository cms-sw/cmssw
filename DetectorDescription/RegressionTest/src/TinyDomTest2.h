#ifndef x_TinyDomTest2_h
#define x_TinyDomTest2_h


#include "DetectorDescription/DDRegressionTest/src/TinyDom2.h"
#include <vector>
using std::vector;

/** some tests for TinyDom and TinyDomWalker */
class TinyDomTest2
{
public:
  explicit TinyDomTest2(const TinyDom2 & td2);
  
  unsigned int allNodes(const Node2 & n2, vector<const AttList2*> & at2);
  
private:
   const TinyDom2 & dom_;

};
#endif 

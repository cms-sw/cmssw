#ifndef x_TinyDomTest_h
#define x_TinyDomTest_h


#include "DetectorDescription/DDRegressionTest/src/TinyDom.h"
#include <vector>
using std::vector;

/** some tests for TinyDom and TinyDomWalker */
class TinyDomTest
{
public:
  explicit TinyDomTest(const TinyDom &);
  
  unsigned int allNodes(const NodeName &, vector<const AttList*> &);
  
private:
   const TinyDom & dom_;
};
#endif 

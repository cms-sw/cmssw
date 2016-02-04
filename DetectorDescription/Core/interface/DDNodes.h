#ifndef DDNodes_h
#define DDNodes_h

#include <vector>
#include "DetectorDescription/Core/interface/DDExpandedView.h"

class DDNodes : public std::vector<DDExpandedNode>
{
public:
  DDNodes()  { }
  ~DDNodes() { }
};

#endif

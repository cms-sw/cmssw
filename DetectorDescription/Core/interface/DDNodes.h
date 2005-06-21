#ifndef DDNodes_h
#define DDNodes_h

#include <vector>
#include "DetectorDescription/DDCore/interface/DDExpandedView.h"

class DDNodes : public vector<DDExpandedNode>
{
public:
  DDNodes()  { }
  ~DDNodes() { }
};

#endif

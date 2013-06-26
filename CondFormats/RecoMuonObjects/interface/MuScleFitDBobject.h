#ifndef MuScleFitDBobject_h
#define MuScleFitDBobject_h

#include <vector>

struct MuScleFitDBobject
{
  std::vector<int> identifiers;
  std::vector<double> parameters;
  std::vector<double> fitQuality;
};

#endif // MuScleFitDBobject

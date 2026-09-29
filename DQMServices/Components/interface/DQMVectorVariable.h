#ifndef DQMServices_Components_interface_DQMVectorVariable_h
#define DQMServices_Components_interface_DQMVectorVariable_h

#include <functional>
#include <string>
#include <vector>

// Describes a vector-valued quantity extracted from an object of type T
// (e.g. one float per track/crystal associated to the object), to be booked
// as a single 1D histogram by GenericObjectDQMSource<T> and filled once per
// element on every event -- so an electron with 3 associated tracks
// contributes 3 entries to e.g. the "trkpt" histogram, not one.
//
// This is the vector-valued counterpart to DQMVariable<T>; use it for any
// std::vector<...>-typed data member instead of DQMVariable<T>.
template <typename T>
struct DQMVectorVariable {
  std::string name;
  std::string title;
  int nbins;
  double xmin;
  double xmax;
  std::function<std::vector<double>(T const&)> accessor;
};

#endif

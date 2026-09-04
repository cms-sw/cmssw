#ifndef DQMServices_Components_interface_DQMVariable_h
#define DQMServices_Components_interface_DQMVariable_h

#include <functional>
#include <string>

// Describes a single scalar quantity extracted from an object of type T,
// to be booked and filled as a 1D histogram by GenericObjectDQMSource<T>.
//
// One DQMVariable<T> == one histogram. A DQMVariableTraits<T> specialization
// returns the full list of variables to plot for a given T.
template <typename T>
struct DQMVariable {
  std::string name;
  std::string title;
  int nbins;
  double xmin;
  double xmax;
  std::function<double(T const&)> accessor;
};

#endif

#ifndef Framework_demangle_h
#define Framework_demangle_h

#include <boost/core/demangle.hpp>

namespace edm {

  template <typename T>
  inline const std::string demangle = boost::core::demangle(typeid(T).name());

}

#endif  // Framework_demangle_h

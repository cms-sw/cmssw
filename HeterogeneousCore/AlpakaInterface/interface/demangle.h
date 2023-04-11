#ifndef HeterogeneousCore_AlpakaInterface_interface_demangle_h
#define HeterogeneousCore_AlpakaInterface_interface_demangle_h

#include <boost/core/demangle.hpp>

namespace edm {

  template <typename T>
  inline const std::string demangle = boost::core::demangle(typeid(T).name());

}

#endif  // HeterogeneousCore_AlpakaInterface_interface_demangle_h

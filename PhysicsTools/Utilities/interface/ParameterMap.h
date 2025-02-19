#ifndef PhysicsTools_Utilities_ParameterMap_h
#define PhysicsTools_Utilities_ParameterMap_h
#include <map>
#include <string>
#include <vector>

namespace fit {
  struct parameter_t {
    double val, min, max, err;
    bool fixed;
  };
  
  typedef std::map<std::string, parameter_t> parameterMap_t;
  typedef std::vector<std::pair<std::string, parameter_t> > parameterVector_t;
}

#endif

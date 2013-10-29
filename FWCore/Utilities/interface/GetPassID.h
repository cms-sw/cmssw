#ifndef Utilities_GetPassID_h
#define Utilities_GetPassID_h

#include <string>

namespace edm {
  inline
  std::string const getPassID () {
    static std::string const passID;
    // return empty string for now.
    return passID; 
  }
}
#endif

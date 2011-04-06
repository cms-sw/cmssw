#ifndef FWCore_Utilities_ConvertException_h
#define FWCore_Utilities_ConvertException_h

#include <string>
#include <exception>

namespace edm {
  namespace convertException {
    void badAllocToEDM();
    void stdToEDM(std::exception const& e);
    void stringToEDM(std::string& s);
    void charPtrToEDM(char const* c);
    void unknownToEDM();
  }
}

#endif

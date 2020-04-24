#ifndef FWCore_Utilities_ConvertException_h
#define FWCore_Utilities_ConvertException_h

#include <string>
#include <exception>
#include <functional>
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace convertException {
    void badAllocToEDM();
    void stdToEDM(std::exception const& e);
    void stringToEDM(std::string& s);
    void charPtrToEDM(char const* c);
    void unknownToEDM();

    template<typename F>
    auto wrap(F iFunc) -> decltype( iFunc() ) {
      try {
        return iFunc();
      }
      catch (cms::Exception&)  { throw; }
      catch(std::bad_alloc&) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
      //Never gets here
      typedef decltype(iFunc()) ReturnType;      
      return ReturnType();
    }
  }
}

#endif

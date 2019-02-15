#include "PyBind11Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
//#include <iostream>
namespace edm {

  void pythonToCppException(const std::string& iType, const std::string &error)
 {
   throw cms::Exception(iType)<<" unknown python problem occurred.\n"<< error << std::endl;
 }
}



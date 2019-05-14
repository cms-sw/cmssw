#include "FWCore/PyDevParameterSet/interface/PyBind11Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
//#include <iostream>
namespace cmspython3 {

  void pythonToCppException(const std::string& iType, const std::string& error) {
    throw cms::Exception(iType) << " unknown python problem occurred.\n" << error << std::endl;
  }
}  // namespace cmspython3

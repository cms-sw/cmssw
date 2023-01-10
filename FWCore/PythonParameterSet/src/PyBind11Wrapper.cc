#include "FWCore/PythonParameterSet/interface/PyBind11Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
namespace edm {

  void pythonToCppException(const std::string& iType, const std::string& error) {
    auto colon = error.find(':');
    if (colon != std::string::npos) {
      if (error.substr(0, colon) == "EDMException") {
        auto errorNameStart = error.find('{');
        auto errorNameEnd = error.find('}');
        if (errorNameStart != std::string::npos and errorNameEnd != std::string::npos) {
          auto errorName = error.substr(errorNameStart + 1, errorNameEnd - errorNameStart - 1);
          if ("Configuration" == errorName) {
            auto newLine = error.find('\n', errorNameEnd + 1);
            if (newLine == std::string::npos) {
              newLine = errorNameEnd + 1;
            }
            throw edm::Exception(edm::errors::Configuration) << error.substr(newLine + 1, std::string::npos);
          } else if ("UnavailableAccelerator" == errorName) {
            auto newLine = error.find('\n', errorNameEnd + 1);
            if (newLine == std::string::npos) {
              newLine = errorNameEnd + 1;
            }
            throw edm::Exception(edm::errors::UnavailableAccelerator) << error.substr(newLine + 1, std::string::npos);
          }
        }
      }
    }
    throw cms::Exception(iType) << " unknown python problem occurred.\n" << error << std::endl;
  }
}  // namespace edm

#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

namespace edm {

  void convertException::badAllocToEDM() {
    std::cerr << "\nstd::bad_alloc exception" << std::endl;
    edm::Exception e(edm::errors::BadAlloc);
    e << "A std::bad_alloc exception was thrown.\n"
      << "The job has probably exhausted the virtual memory available to the process.\n";
    throw e;
  }

  void convertException::stdToEDM(std::exception const& e) {
    edm::Exception ex(edm::errors::StdException);
    ex << "A std::exception was thrown.\n"
       << e.what();
    throw ex;
  }

  void convertException::stringToEDM(std::string& s) {
    edm::Exception e(edm::errors::BadExceptionType);
    e << "A std::string was thrown as an exception.\n"
      << s;
    throw e;
  }

  void convertException::charPtrToEDM(char const* c) {
    edm::Exception e(edm::errors::BadExceptionType);
    e << "A const char* was thrown as an exception.\n"
      << c;
    throw e;
  }

  void convertException::unknownToEDM() {
    edm::Exception e(edm::errors::Unknown);
    e << "An exception of unknown type was thrown.\n";
    throw e;
  }
}

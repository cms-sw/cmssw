#ifndef FWCore_MessageLogger_ExceptionMessages
#define FWCore_MessageLogger_ExceptionMessages

namespace cms {
  class Exception;
}

namespace edm {
  class JobReport;

  void printCmsException(cms::Exception& e, edm::JobReport * jobRep = nullptr, int rc = -1);
  void printCmsExceptionWarning(char const* behavior, cms::Exception const& e);
}
#endif

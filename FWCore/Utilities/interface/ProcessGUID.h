#ifndef FWCore_Utilities_ProcessGUID_h
#define FWCore_Utilities_ProcessGUID_h

#include <string>

namespace edm {
  /**
   * This class is an abstract base class for a Service providing a
   * process-level Globally Unique Identifier (GUID).
   */
  class ProcessGUID {
  public:
    virtual ~ProcessGUID() = 0;

    virtual std::string binary() const = 0;
    virtual std::string string() const = 0;
  };
}  // namespace edm

#endif

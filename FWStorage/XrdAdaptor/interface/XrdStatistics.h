#ifndef Utilities_XrdAdaptor_XrdStatistics_h
#define Utilities_XrdAdaptor_XrdStatistics_h

#include <vector>
#include <string>
#include <chrono>

namespace xrd_adaptor {

  class XrdStatistics {
  public:
    XrdStatistics() {}
    virtual ~XrdStatistics();

    struct CondorIOStats {
      uint64_t bytesRead{0};
      std::chrono::nanoseconds transferTime{0};
    };

    // Provide an update of per-site transfer statistics to the CondorStatusService.
    // Returns a mapping of "site name" to transfer statistics.  The "site name" is
    // as self-identified by the Xrootd host; may not necessarily match up with the
    // "CMS site name".
    virtual std::vector<std::pair<std::string, CondorIOStats>> condorUpdate() = 0;
  };

}  // namespace xrd_adaptor

#endif

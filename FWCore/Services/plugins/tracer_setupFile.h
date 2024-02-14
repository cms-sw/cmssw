#ifndef FWCore_Services_tracer_setupFile_h
#define FWCore_Services_tracer_setupFile_h

#include <string>

namespace edm {
  class ActivityRegistry;
  namespace service::tracer {
    void setupFile(std::string const& iFileName, edm::ActivityRegistry&);
  }
}  // namespace edm

#endif

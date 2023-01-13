#include "FWCore/Framework/interface/ExceptionHelpers.h"

#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cstring>

namespace edm {

  void addContextAndPrintException(char const* context, cms::Exception& ex, bool disablePrint) {
    if (context != nullptr && strlen(context) != 0U) {
      ex.addContext(context);
    }
    if (!disablePrint) {
      Service<JobReport> jobReportSvc;
      if (jobReportSvc.isAvailable()) {
        JobReport* jobRep = jobReportSvc.operator->();
        edm::printCmsException(ex, jobRep, ex.returnCode());
      } else {
        edm::printCmsException(ex);
      }
      ex.setAlreadyPrinted();
    }
  }
}  // namespace edm

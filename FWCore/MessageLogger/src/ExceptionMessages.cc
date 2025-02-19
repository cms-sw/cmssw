#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <string>
#include <sstream>
#include <iomanip>

namespace edm {
  void 
  printCmsException(cms::Exception& e, edm::JobReport * jobRep, int rc) try {
    std::string shortDesc("Fatal Exception");
    std::ostringstream longDesc;
    longDesc << e.explainSelf();
    LogAbsolute(shortDesc)
      << "----- Begin " << shortDesc << " "
      << std::setprecision(0) << TimeOfDay()
      << "-----------------------\n"
      << longDesc.str()
      << "----- End " << shortDesc << " -------------------------------------------------";
    if(jobRep) jobRep->reportError(shortDesc, longDesc.str(), rc);
  } catch(...) {
  }

  void
  printCmsExceptionWarning(char const* behavior, cms::Exception const& e, edm::JobReport * jobRep, int rc) try {
    std::string shortDesc(behavior);
    shortDesc += " Exception";
    std::ostringstream longDesc;
    longDesc << e.explainSelf();
    LogPrint(shortDesc)
      << "----- Begin " << shortDesc << " "
      << std::setprecision(0) << TimeOfDay()
      << "-----------------------\n"
      << longDesc.str()
      << "----- End " << shortDesc << " -------------------------------------------------";
    if(jobRep) jobRep->reportError(shortDesc, longDesc.str(), rc);
  } catch(...) {
  }
}

#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <string>
#include <sstream>

namespace edm {
  void 
  printCmsException(cms::Exception& e, char const* prog, edm::JobReport * jobRep, int rc) try {
    std::string programName(prog ? prog : "program");
    std::string shortDesc("CMSException");
    std::ostringstream longDesc;
    longDesc << "cms::Exception caught in " 
	     << programName
	     << "\n"
	     << e.explainSelf();
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
    if(jobRep) jobRep->reportError(shortDesc, longDesc.str(), rc);
  } catch(...) {
  }

  void printBadAllocException(char const *prog, edm::JobReport * jobRep, int rc) try {
    std::string programName(prog ? prog : "program");
    std::string shortDesc("std::bad_allocException");
    std::ostringstream longDesc;
    longDesc << "std::bad_alloc exception caught in "
	     << programName
	     << "\n"
	     << "The job has probably exhausted the virtual memory available to the process.\n";
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
    if(jobRep) jobRep->reportError(shortDesc, longDesc.str(), rc);
  } catch(...) {
  }

  void printStdException(std::exception& e, char const*prog, edm::JobReport * jobRep, int rc) try {
    std::string programName(prog ? prog : "program");
    std::string shortDesc("StdLibException");
    std::ostringstream longDesc;
    longDesc << "Standard library exception caught in " 
	     << programName
	     << "\n"
	     << e.what();
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
    if (jobRep) jobRep->reportError(shortDesc, longDesc.str(), rc);
  } catch(...) {
  }

  void printUnknownException(char const *prog, edm::JobReport * jobRep, int rc) try {
    std::string programName(prog ? prog : "program");
    std::string shortDesc("UnknownException");
    std::ostringstream longDesc;
    longDesc << "Unknown exception caught in "
	     << programName
	     << "\n";
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
    if (jobRep) jobRep->reportError(shortDesc, longDesc.str(), rc);
  } catch(...) {
  }
}

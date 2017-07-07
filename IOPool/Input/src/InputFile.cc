/*----------------------------------------------------------------------
Holder for an input TFile.
----------------------------------------------------------------------*/
#include "TList.h"
#include "TStreamerInfo.h"
#include "TClass.h"
#include "InputFile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ExceptionPropagate.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <exception>
#include <iomanip>

namespace edm {
  InputFile::InputFile(char const* fileName, char const* msg, InputType inputType) :
    file_(), fileName_(fileName), reportToken_(0), inputType_(inputType) {

    // ROOT's context management implicitly assumes that a file is opened and
    // closed on the same thread.  To avoid the problem, we declare a local
    // TContext object; when it goes out of scope, its destructor unregisters
    // the context, guaranteeing the context is unregistered in the same thread
    // it was registered in.  Fixes issue #15524.
    TDirectory::TContext contextEraser;

    logFileAction(msg, fileName);
    file_ = std::unique_ptr<TFile>(TFile::Open(fileName)); // propagate_const<T> has no reset() function
    std::exception_ptr e = edm::threadLocalException::getException();
    if(e != std::exception_ptr()) {
      edm::threadLocalException::setException(std::exception_ptr());
      std::rethrow_exception(e);
    }
    if(!file_) {
      return;
    }
    if(file_->IsZombie()) {
      file_ = nullptr; // propagate_const<T> has no reset() function
      return;
    }
    
    logFileAction("  Successfully opened file ", fileName);
  }

  InputFile::~InputFile() {
    Close();
  }

  void
  InputFile::inputFileOpened(std::string const& logicalFileName,
                             std::string const& inputType,
                             std::string const& moduleName,
                             std::string const& label,
                             std::string const& fid,
                             std::vector<std::string> const& branchNames) {
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->inputFileOpened(fileName_,
                                              logicalFileName,
                                              std::string(),
                                              inputType,
                                              moduleName,
                                              label,
                                              fid,
                                              branchNames);
  }

  void
  InputFile::eventReadFromFile() const {
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(inputType_, reportToken_);
  }

  void
  InputFile::reportInputRunNumber(unsigned int run) const {
    Service<JobReport> reportSvc;
    reportSvc->reportInputRunNumber(run);
  }

  void
  InputFile::reportInputLumiSection(unsigned int run, unsigned int lumi) const {
    Service<JobReport> reportSvc;
    reportSvc->reportInputLumiSection(run, lumi);
  }

  void
  InputFile::reportSkippedFile(std::string const& fileName, std::string const& logicalFileName) {
    Service<JobReport> reportSvc;
    reportSvc->reportSkippedFile(fileName, logicalFileName);
  }

  void
  InputFile::reportFallbackAttempt(std::string const& pfn, std::string const& logicalFileName, std::string const& errorMessage) {
    Service<JobReport> reportSvc;
    reportSvc->reportFallbackAttempt(pfn, logicalFileName, errorMessage);
  }

  void
  InputFile::Close() {
    if(file_->IsOpen()) {
      file_->Close();
      try {
        logFileAction("  Closed file ", fileName_.c_str());
        Service<JobReport> reportSvc;
        reportSvc->inputFileClosed(inputType_, reportToken_);
      } catch(std::exception) {
        // If Close() called in a destructor after an exception throw, the services may no longer be active.
        // Therefore, we catch any reasonable new exception.
      }
    }
  }

  void
  InputFile::logFileAction(char const* msg, char const* fileName) const {
    LogAbsolute("fileAction") << std::setprecision(0) << TimeOfDay() << msg << fileName;
    FlushMessageLog();
  }

  void
  InputFile::reportReadBranches() {
    Service<JobReport> reportSvc;
    reportSvc->reportReadBranches();
  }

  void
  InputFile::reportReadBranch(InputType inputType, std::string const& branchName) {
    Service<JobReport> reportSvc;
    reportSvc->reportReadBranch(inputType, branchName);
  }
}

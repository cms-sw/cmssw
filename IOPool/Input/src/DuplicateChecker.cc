
#include "IOPool/Input/src/DuplicateChecker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <cassert>
#include <algorithm>

namespace edm {

  DuplicateChecker::DuplicateChecker(ParameterSet const& pset) :
    dataType_(unknown),
    itIsKnownTheFileHasNoDuplicates_(false)
  {
    // The default value provided as the second argument to the getUntrackedParameter function call
    // is not used when the ParameterSet has been validated and the parameters are not optional
    // in the description.  This is currently true when PoolSource is the primary input source.
    // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
    // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
    // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
    // and should be deleted from the code.
    std::string duplicateCheckMode =
      pset.getUntrackedParameter<std::string>("duplicateCheckMode", std::string("checkAllFilesOpened"));

    if (duplicateCheckMode == std::string("noDuplicateCheck")) duplicateCheckMode_ = noDuplicateCheck;
    else if (duplicateCheckMode == std::string("checkEachFile")) duplicateCheckMode_ = checkEachFile;
    else if (duplicateCheckMode == std::string("checkEachRealDataFile")) duplicateCheckMode_ = checkEachRealDataFile;
    else if (duplicateCheckMode == std::string("checkAllFilesOpened")) duplicateCheckMode_ = checkAllFilesOpened;
    else {
      throw cms::Exception("Configuration")
        << "Illegal configuration parameter value passed to PoolSource for\n"
        << "the \"duplicateCheckMode\" parameter, legal values are:\n"
        << "\"noDuplicateCheck\", \"checkEachFile\", \"checkEachRealDataFile\", \"checkAllFilesOpened\"\n";
    }
  }

  void DuplicateChecker::inputFileOpened(
      bool realData,
      IndexIntoFile const& indexIntoFile,
      std::vector<boost::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
      std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile) {

    dataType_ = realData ? isRealData : isSimulation;
    if (checkDisabled()) return;

    relevantPreviousEvents_.clear();
    itIsKnownTheFileHasNoDuplicates_ = false;

    if (duplicateCheckMode_ == checkAllFilesOpened) {

      // Compares the current IndexIntoFile to all the previous ones and saves any duplicates.
      // One unintended thing, it also saves the duplicate runs and lumis.
      for(std::vector<boost::shared_ptr<IndexIntoFile> >::size_type i = 0; i < currentIndexIntoFile; ++i) {
        if (indexesIntoFiles[i].get() != 0) {

          indexIntoFile.set_intersection(*indexesIntoFiles[i], relevantPreviousEvents_);
        }
      }
    }
    if (relevantPreviousEvents_.empty()) {
      if(!indexIntoFile.containsDuplicateEvents()) {
        itIsKnownTheFileHasNoDuplicates_ = true;
      }
    }
  }

  void DuplicateChecker::inputFileClosed()
  {
    dataType_ = unknown;
    relevantPreviousEvents_.clear();
    itIsKnownTheFileHasNoDuplicates_ = false;
  }

  bool DuplicateChecker::isDuplicateAndCheckActive(int index,
                                                   RunNumber_t run,
                                                   LuminosityBlockNumber_t lumi,
                                                   EventNumber_t event,
                                                   std::string const& fileName) {
    if (itIsKnownTheFileHasNoDuplicates_) return false;
    if (checkDisabled()) return false;

    IndexIntoFile::IndexRunLumiEventKey newEvent(index, run, lumi, event);
    bool duplicate = !relevantPreviousEvents_.insert(newEvent).second;

    if (duplicate) {
      if (duplicateCheckMode_ == checkAllFilesOpened) {
        LogWarning("DuplicateEvent")
          << "Duplicate Events found in entire set of input files.\n"
          << "Both events were from run " << run 
          << " and luminosity block " << lumi
          << " with event number " << event << ".\n"
          << "The duplicate was from file " << fileName << ".\n"
          << "The duplicate will be skipped.\n";
      }
      else {
        LogWarning("DuplicateEvent")
          << "Duplicate Events found in file " << fileName << ".\n"
          << "Both events were from run " << run
          << " and luminosity block " << lumi
          << " with event number " << event << ".\n"
          << "The duplicate will be skipped.\n";
      }
      return true;
    }
    return false;
  }

  void
  DuplicateChecker::fillDescription(ParameterSetDescription & desc) {
    std::string defaultString("checkAllFilesOpened");
    desc.addUntracked<std::string>("duplicateCheckMode", defaultString);
  }
}

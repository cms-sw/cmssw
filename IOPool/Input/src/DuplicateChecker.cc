
#include "IOPool/Input/src/DuplicateChecker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cassert>
#include <algorithm>

namespace edm {

  DuplicateChecker::DuplicateChecker(ParameterSet const& pset) :
    dataType_(unknown),
    itIsKnownTheFileHasNoDuplicates_(false)
  {
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
      FileIndex const& fileIndex,
      std::vector<boost::shared_ptr<FileIndex> > const& fileIndexes,
      std::vector<boost::shared_ptr<FileIndex> >::size_type currentFileIndex) {

    if (duplicateCheckMode_ == noDuplicateCheck) return;

    dataType_ = realData ? isRealData : isSimulation;
    if (duplicateCheckMode_ == checkEachRealDataFile) {
      if(dataType_ == isSimulation) return;
    }

    relevantPreviousEvents_.clear();
    itIsKnownTheFileHasNoDuplicates_ = false;

    if (duplicateCheckMode_ == checkAllFilesOpened) {

      std::insert_iterator<std::set<FileIndex::Element> > insertIter(relevantPreviousEvents_, relevantPreviousEvents_.begin());

      // Compares the current FileIndex to all the previous ones and saves any duplicates.
      // One unintended thing, it also saves the duplicate runs and lumis, but this should not cause any problems
      for(std::vector<boost::shared_ptr<FileIndex> >::size_type i = 0; i < currentFileIndex; ++i) {
        if (fileIndexes[i].get() != 0) {
          std::set_intersection(fileIndex.begin(), fileIndex.end(),
                                fileIndexes[i]->begin(), fileIndexes[i]->end(),
                                insertIter);
        }
      }
    }
    if (relevantPreviousEvents_.empty()) {
      FileIndex::const_iterator duplicate = std::adjacent_find(fileIndex.begin(), fileIndex.end());
      if (duplicate == fileIndex.end()) {
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

  bool DuplicateChecker::fastCloningOK() const
  {
    return 
      itIsKnownTheFileHasNoDuplicates_ ||
      duplicateCheckMode_ == noDuplicateCheck ||
      (duplicateCheckMode_ == checkEachRealDataFile && dataType_ == isSimulation);
  }

  bool DuplicateChecker::isDuplicateAndCheckActive(EventID const& eventID,
                                                   LuminosityBlockNumber_t const& lumi,
                                                   std::string const& fileName)
  {
    if (itIsKnownTheFileHasNoDuplicates_) return false;
    if (duplicateCheckMode_ == noDuplicateCheck) return false;
    if (duplicateCheckMode_ == checkEachRealDataFile) {
      assert(dataType_ != unknown);
      if(dataType_ == isSimulation) return false;
    }

    FileIndex::Element newEvent(eventID.run(), lumi, eventID.event());
    bool duplicate = !relevantPreviousEvents_.insert(newEvent).second;

    if (duplicate) {
      if (duplicateCheckMode_ == checkAllFilesOpened) {
        LogWarning("DuplicateEvent")
          << "Duplicate Events found in entire set of input files.\n"
          << "Both events were from run " << eventID.run() 
          << " and luminosity block " << lumi 
          << " with event number " << eventID.event() << ".\n"
          << "The duplicate was from file " << fileName << ".\n"
          << "The duplicate will be skipped.\n";
      }
      else {
        LogWarning("DuplicateEvent")
          << "Duplicate Events found in file " << fileName << ".\n"
          << "Both events were from run " << eventID.run() 
          << " and luminosity block " << lumi 
          << " with event number " << eventID.event() << ".\n"
          << "The duplicate will be skipped.\n";
      }
      return true;
    }
    return false;
  }
}

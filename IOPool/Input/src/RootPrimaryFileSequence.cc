/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DuplicateChecker.h"
#include "InputFile.h"
#include "PoolSource.h"
#include "RootFile.h"
#include "RootPrimaryFileSequence.h"
#include "RootTree.h"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

namespace edm {
  RootPrimaryFileSequence::RootPrimaryFileSequence(ParameterSet const& pset,
                                                   PoolSource& input,
                                                   InputFileCatalog const& catalog)
      : RootInputFileSequence(pset, catalog),
        input_(input),
        firstFile_(true),
        branchesMustMatch_(BranchDescription::Permissive),
        orderedProcessHistoryIDs_(),
        eventSkipperByID_(EventSkipperByID::create(pset).release()),
        initialNumberOfEventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents")),
        noRunLumiSort_(pset.getUntrackedParameter<bool>("noRunLumiSort")),
        noEventSort_(noRunLumiSort_ ? true : pset.getUntrackedParameter<bool>("noEventSort")),
        treeCacheSize_(noEventSort_ ? pset.getUntrackedParameter<unsigned int>("cacheSize") : 0U),
        duplicateChecker_(new DuplicateChecker(pset)),
        usingGoToEvent_(false),
        enablePrefetching_(false),
        enforceGUIDInFileName_(pset.getUntrackedParameter<bool>("enforceGUIDInFileName")) {
    if (noRunLumiSort_ && (remainingEvents() >= 0 || remainingLuminosityBlocks() >= 0)) {
      // There would need to be some Framework development work to allow stopping
      // early with noRunLumiSort set true related to closing lumis and runs that
      // were supposed to be continued but were not... We cannot have events written
      // to output with no run or lumi written to output.
      throw Exception(errors::Configuration,
                      "Illegal to configure noRunLumiSort and limit the number of events or luminosityBlocks");
    }
    // The SiteLocalConfig controls the TTreeCache size and the prefetching settings.
    Service<SiteLocalConfig> pSLC;
    if (pSLC.isAvailable()) {
      if (treeCacheSize_ != 0U && pSLC->sourceTTreeCacheSize()) {
        treeCacheSize_ = *(pSLC->sourceTTreeCacheSize());
      }
      enablePrefetching_ = pSLC->enablePrefetching();
    }

    std::string branchesMustMatch =
        pset.getUntrackedParameter<std::string>("branchesMustMatch", std::string("permissive"));
    if (branchesMustMatch == std::string("strict"))
      branchesMustMatch_ = BranchDescription::Strict;

    // Prestage the files
    for (setAtFirstFile(); !noMoreFiles(); setAtNextFile()) {
      storage::StorageFactory::get()->stagein(fileNames()[0]);
    }
    // Open the first file.
    for (setAtFirstFile(); !noMoreFiles(); setAtNextFile()) {
      initFile(input_.skipBadFiles());
      if (rootFile())
        break;
    }
    if (rootFile()) {
      input_.productRegistryUpdate().updateFromInput(rootFile()->productRegistry()->productList());
      if (initialNumberOfEventsToSkip_ != 0) {
        skipEventsAtBeginning(initialNumberOfEventsToSkip_);
      }
    }
  }

  RootPrimaryFileSequence::~RootPrimaryFileSequence() {}

  void RootPrimaryFileSequence::endJob() { closeFile(); }

  std::shared_ptr<FileBlock> RootPrimaryFileSequence::readFile_() {
    std::shared_ptr<FileBlock> fileBlock;
    if (firstFile_) {
      firstFile_ = false;
      // Usually the first input file will already be open
      if (!rootFile()) {
        initFile(input_.skipBadFiles());
      }
    } else if (goToEventInNewFile_) {
      goToEventInNewFile_ = false;
      setAtFileSequenceNumber(goToFileSequenceOffset_);
      initFile(false);
      assert(rootFile());
      bool found = rootFile()->goToEvent(goToEventID_);
      assert(found);
    } else if (skipIntoNewFile_) {
      skipIntoNewFile_ = false;
      setAtFileSequenceNumber(skipToFileSequenceNumber_);
      initFile(false);
      assert(rootFile());
      if (skipToOffsetInFinalFile_ < 0) {
        rootFile()->setToLastEntry();
      }
      bool atEnd = rootFile()->skipEvents(skipToOffsetInFinalFile_);
      assert(!atEnd && skipToOffsetInFinalFile_ == 0);
    } else {
      if (!nextFile()) {
        // handle case with last file bad and
        // skipBadFiles true
        fb_ = fileBlock;
        return fileBlock;
      }
    }
    if (!rootFile()) {
      fileBlock = std::make_shared<FileBlock>();
      fb_ = fileBlock;
      return fileBlock;
    }
    fileBlock = rootFile()->createFileBlock();
    fb_ = fileBlock;
    return fileBlock;
  }

  void RootPrimaryFileSequence::closeFile_() {
    // close the currently open file, if any, and delete the RootFile object.
    if (rootFile()) {
      auto sentry = std::make_unique<InputSource::FileCloseSentry>(input_, lfn());
      rootFile()->close();
      if (duplicateChecker_)
        duplicateChecker_->inputFileClosed();
      rootFile().reset();
    }
  }

  void RootPrimaryFileSequence::initFile_(bool skipBadFiles) {
    // If we are not duplicate checking across files and we are not using random access to find events,
    // then we can delete the IndexIntoFile for the file we are closing.
    // If we can't delete all of it, then we can delete the parts we do not need.
    bool deleteIndexIntoFile = !usingGoToEvent_ && !(duplicateChecker_ && duplicateChecker_->checkingAllFiles() &&
                                                     !duplicateChecker_->checkDisabled());
    initTheFile(skipBadFiles, deleteIndexIntoFile, &input_, "primaryFiles", InputType::Primary);
  }

  RootPrimaryFileSequence::RootFileSharedPtr RootPrimaryFileSequence::makeRootFile(std::shared_ptr<InputFile> filePtr) {
    size_t currentIndexIntoFile = sequenceNumberOfFile();
    return std::make_shared<RootFile>(fileNames()[0],
                                      input_.processConfiguration(),
                                      logicalFileName(),
                                      filePtr,
                                      eventSkipperByID(),
                                      initialNumberOfEventsToSkip_ != 0,
                                      remainingEvents(),
                                      remainingLuminosityBlocks(),
                                      input_.nStreams(),
                                      treeCacheSize_,
                                      input_.treeMaxVirtualSize(),
                                      input_.processingMode(),
                                      input_.runHelper(),
                                      noRunLumiSort_,
                                      noEventSort_,
                                      input_.productSelectorRules(),
                                      InputType::Primary,
                                      input_.branchIDListHelper(),
                                      input_.processBlockHelper().get(),
                                      input_.thinnedAssociationsHelper(),
                                      nullptr,  // associationsFromSecondary
                                      duplicateChecker(),
                                      input_.dropDescendants(),
                                      input_.processHistoryRegistryForUpdate(),
                                      indexesIntoFiles(),
                                      currentIndexIntoFile,
                                      orderedProcessHistoryIDs_,
                                      input_.bypassVersionCheck(),
                                      input_.labelRawDataLikeMC(),
                                      usingGoToEvent_,
                                      enablePrefetching_,
                                      enforceGUIDInFileName_);
  }

  bool RootPrimaryFileSequence::nextFile() {
    do {
      if (!noMoreFiles())
        setAtNextFile();
      if (noMoreFiles()) {
        return false;
      }

      initFile(input_.skipBadFiles());
      if (rootFile()) {
        break;
      }
      // If we are not skipping bad files and the file
      // open failed, then initFile should have thrown
      assert(input_.skipBadFiles());
    } while (true);

    // make sure the new product registry is compatible with the main one
    std::string mergeInfo =
        input_.productRegistryUpdate().merge(*rootFile()->productRegistry(), fileNames()[0], branchesMustMatch_);
    if (!mergeInfo.empty()) {
      throw Exception(errors::MismatchedInputFiles, "RootPrimaryFileSequence::nextFile()") << mergeInfo;
    }
    return true;
  }

  bool RootPrimaryFileSequence::previousFile() {
    if (atFirstFile()) {
      return false;
    }
    setAtPreviousFile();

    initFile(false);

    if (rootFile()) {
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo =
          input_.productRegistryUpdate().merge(*rootFile()->productRegistry(), fileNames()[0], branchesMustMatch_);
      if (!mergeInfo.empty()) {
        throw Exception(errors::MismatchedInputFiles, "RootPrimaryFileSequence::previousEvent()") << mergeInfo;
      }
    }
    if (rootFile())
      rootFile()->setToLastEntry();
    return true;
  }

  InputSource::ItemType RootPrimaryFileSequence::getNextItemType(RunNumber_t& run,
                                                                 LuminosityBlockNumber_t& lumi,
                                                                 EventNumber_t& event) {
    if (noMoreFiles() || skipToStop_) {
      skipToStop_ = false;
      return InputSource::IsStop;
    }
    if (firstFile_ || goToEventInNewFile_ || skipIntoNewFile_) {
      return InputSource::IsFile;
    }
    if (rootFile()) {
      IndexIntoFile::EntryType entryType = rootFile()->getNextItemType(run, lumi, event);
      if (entryType == IndexIntoFile::kEvent) {
        return InputSource::IsEvent;
      } else if (entryType == IndexIntoFile::kLumi) {
        return InputSource::IsLumi;
      } else if (entryType == IndexIntoFile::kRun) {
        return InputSource::IsRun;
      }
      assert(entryType == IndexIntoFile::kEnd);
    }
    if (atLastFile()) {
      return InputSource::IsStop;
    }
    return InputSource::IsFile;
  }

  // Rewind to before the first event that was read.
  void RootPrimaryFileSequence::rewind_() {
    if (!atFirstFile()) {
      closeFile();
      setAtFirstFile();
    }
    if (!rootFile()) {
      initFile(false);
    }
    rewindFile();
    firstFile_ = true;
    goToEventInNewFile_ = false;
    skipIntoNewFile_ = false;
    skipToStop_ = false;
    if (rootFile()) {
      if (initialNumberOfEventsToSkip_ != 0) {
        skipEventsAtBeginning(initialNumberOfEventsToSkip_);
      }
    }
  }

  // Rewind to the beginning of the current file
  void RootPrimaryFileSequence::rewindFile() {
    if (rootFile())
      rootFile()->rewind();
  }

  // Advance "offset" events.  Offset will be positive.
  void RootPrimaryFileSequence::skipEventsAtBeginning(int offset) {
    assert(rootFile());
    assert(offset >= 0);
    while (offset != 0) {
      bool atEnd = rootFile()->skipEvents(offset);
      if ((offset > 0 || atEnd) && !nextFile()) {
        return;
      }
    }
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void RootPrimaryFileSequence::skipEvents(int offset) {
    assert(rootFile());

    bool atEnd = rootFile()->skipEvents(offset);
    if (!atEnd && offset == 0) {
      // successfully completed skip within current file
      return;
    }

    // Return, if without closing the current file we know the skip cannot be completed
    skipToStop_ = false;
    if (offset > 0 || atEnd) {
      if (atLastFile() || noMoreFiles()) {
        skipToStop_ = true;
        return;
      }
    }
    if (offset < 0 && atFirstFile()) {
      skipToStop_ = true;
      return;
    }

    // Save the current file and position so that we can restore them
    size_t const originalFileSequenceNumber = sequenceNumberOfFile();
    IndexIntoFile::IndexIntoFileItr originalPosition = rootFile()->indexIntoFileIter();

    if ((offset > 0 || atEnd) && !nextFile()) {
      skipToStop_ = true;  // Can only get here if skipBadFiles is true
    }
    if (offset < 0 && !previousFile()) {
      skipToStop_ = true;  // Can't actually get here
    }

    if (!skipToStop_) {
      while (offset != 0) {
        skipToOffsetInFinalFile_ = offset;
        bool atEnd = rootFile()->skipEvents(offset);
        if ((offset > 0 || atEnd) && !nextFile()) {
          skipToStop_ = true;
          break;
        }
        if (offset < 0 && !previousFile()) {
          skipToStop_ = true;
          break;
        }
      }
      if (!skipToStop_) {
        skipIntoNewFile_ = true;
      }
    }
    skipToFileSequenceNumber_ = sequenceNumberOfFile();

    // Restore the original file and position
    setAtFileSequenceNumber(originalFileSequenceNumber);
    initFile(false);
    assert(rootFile());
    rootFile()->setPosition(originalPosition);
    rootFile()->updateFileBlock(*fb_);
  }

  bool RootPrimaryFileSequence::goToEvent(EventID const& eventID) {
    usingGoToEvent_ = true;
    if (rootFile()) {
      if (rootFile()->goToEvent(eventID)) {
        return true;
      }
      // If only one input file, give up now, to save time.
      if (rootFile() && indexesIntoFiles().size() == 1) {
        return false;
      }
      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      for (auto it = indexesIntoFiles().begin(), itEnd = indexesIntoFiles().end(); it != itEnd; ++it) {
        if (*it && (*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
          goToEventInNewFile_ = true;
          goToFileSequenceOffset_ = it - indexesIntoFiles().begin();
          goToEventID_ = eventID;
          return true;
        }
      }

      // Save the current file and position so that we can restore them
      bool closedOriginalFile = false;
      size_t const originalFileSequenceNumber = sequenceNumberOfFile();
      IndexIntoFile::IndexIntoFileItr originalPosition = rootFile()->indexIntoFileIter();

      // Look for item in files not yet opened.
      bool foundIt = false;
      for (auto it = indexesIntoFiles().begin(), itEnd = indexesIntoFiles().end(); it != itEnd; ++it) {
        if (!*it) {
          setAtFileSequenceNumber(it - indexesIntoFiles().begin());
          initFile(false);
          assert(rootFile());
          closedOriginalFile = true;
          if ((*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
            foundIt = true;
            goToEventInNewFile_ = true;
            goToFileSequenceOffset_ = it - indexesIntoFiles().begin();
            goToEventID_ = eventID;
          }
        }
      }
      if (closedOriginalFile) {
        setAtFileSequenceNumber(originalFileSequenceNumber);
        initFile(false);
        assert(rootFile());
        rootFile()->setPosition(originalPosition);
        rootFile()->updateFileBlock(*fb_);
      }
      return foundIt;
    }
    return false;
  }

  int RootPrimaryFileSequence::remainingEvents() const { return input_.remainingEvents(); }

  int RootPrimaryFileSequence::remainingLuminosityBlocks() const { return input_.remainingLuminosityBlocks(); }

  void RootPrimaryFileSequence::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<unsigned int>("skipEvents", 0U)
        ->setComment("Skip the first 'skipEvents' events that otherwise would have been processed.");
    desc.addUntracked<bool>("noEventSort", true)
        ->setComment(
            "True:  Process runs, lumis and events in the order they appear in the file (but see notes 1 and 2).\n"
            "False: Process runs, lumis and events in each file in numerical order (run#, lumi#, event#) (but see note "
            "3).\n"
            "Note 1: Events within the same lumi will always be processed contiguously.\n"
            "Note 2: Lumis within the same run will always be processed contiguously.\n"
            "Note 3: Any sorting occurs independently in each input file (no sorting across input files).");
    desc.addUntracked<bool>("noRunLumiSort", false)
        ->setComment(
            "True:  Process runs, lumis and events in the order they appear in the file.\n"
            "False: Follow settings based on 'noEventSort' setting.");
    desc.addUntracked<unsigned int>("cacheSize", roottree::defaultCacheSize)
        ->setComment("Size of ROOT TTree prefetch cache.  Affects performance.");
    std::string defaultString("permissive");
    desc.addUntracked<std::string>("branchesMustMatch", defaultString)
        ->setComment(
            "'strict':     Branches in each input file must match those in the first file.\n"
            "'permissive': Branches in each input file may be any subset of those in the first file.");
    desc.addUntracked<bool>("enforceGUIDInFileName", false)
        ->setComment(
            "True:  file name part is required to be equal to the GUID of the file\n"
            "False: file name can be anything");

    EventSkipperByID::fillDescription(desc);
    DuplicateChecker::fillDescription(desc);
  }

  ProcessingController::ForwardState RootPrimaryFileSequence::forwardState() const {
    if (rootFile()) {
      if (!rootFile()->wasLastEventJustRead()) {
        return ProcessingController::kEventsAheadInFile;
      }
      if (noMoreFiles() || atLastFile()) {
        return ProcessingController::kAtLastEvent;
      } else {
        return ProcessingController::kNextFileExists;
      }
    }
    return ProcessingController::kUnknownForward;
  }

  ProcessingController::ReverseState RootPrimaryFileSequence::reverseState() const {
    if (rootFile()) {
      if (!rootFile()->wasFirstEventJustRead()) {
        return ProcessingController::kEventsBackwardsInFile;
      }
      if (!atFirstFile()) {
        return ProcessingController::kPreviousFileExists;
      }
      return ProcessingController::kAtFirstEvent;
    }
    return ProcessingController::kUnknownReverse;
  }

}  // namespace edm

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
        noEventSort_(pset.getUntrackedParameter<bool>("noEventSort")),
        treeCacheSize_(noEventSort_ ? pset.getUntrackedParameter<unsigned int>("cacheSize") : 0U),
        duplicateChecker_(new DuplicateChecker(pset)),
        usingGoToEvent_(false),
        enablePrefetching_(false),
        enforceGUIDInFileName_(pset.getUntrackedParameter<bool>("enforceGUIDInFileName")) {
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
      StorageFactory::get()->stagein(fileName());
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
        skipEvents(initialNumberOfEventsToSkip_);
      }
    }
  }

  RootPrimaryFileSequence::~RootPrimaryFileSequence() {}

  void RootPrimaryFileSequence::endJob() { closeFile_(); }

  std::unique_ptr<FileBlock> RootPrimaryFileSequence::readFile_() {
    if (firstFile_) {
      // The first input file has already been opened.
      firstFile_ = false;
      if (!rootFile()) {
        initFile(input_.skipBadFiles());
      }
    } else {
      if (!nextFile()) {
        assert(0);
      }
    }
    if (!rootFile()) {
      return std::make_unique<FileBlock>();
    }
    return rootFile()->createFileBlock();
  }

  void RootPrimaryFileSequence::closeFile_() {
    // close the currently open file, if any, and delete the RootFile object.
    if (rootFile()) {
      auto sentry = std::make_unique<InputSource::FileCloseSentry>(input_, lfn(), usedFallback());
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
    return std::make_shared<RootFile>(fileName(),
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
                                      noEventSort_,
                                      input_.productSelectorRules(),
                                      InputType::Primary,
                                      input_.branchIDListHelper(),
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
    if (!noMoreFiles())
      setAtNextFile();
    if (noMoreFiles()) {
      return false;
    }

    initFile(input_.skipBadFiles());

    if (rootFile()) {
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo =
          input_.productRegistryUpdate().merge(*rootFile()->productRegistry(), fileName(), branchesMustMatch_);
      if (!mergeInfo.empty()) {
        throw Exception(errors::MismatchedInputFiles, "RootPrimaryFileSequence::nextFile()") << mergeInfo;
      }
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
          input_.productRegistryUpdate().merge(*rootFile()->productRegistry(), fileName(), branchesMustMatch_);
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
    if (noMoreFiles()) {
      return InputSource::IsStop;
    }
    if (firstFile_) {
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
      closeFile_();
      setAtFirstFile();
    }
    if (!rootFile()) {
      initFile(false);
    }
    rewindFile();
    firstFile_ = true;
    if (rootFile()) {
      if (initialNumberOfEventsToSkip_ != 0) {
        skipEvents(initialNumberOfEventsToSkip_);
      }
    }
  }

  // Rewind to the beginning of the current file
  void RootPrimaryFileSequence::rewindFile() {
    if (rootFile())
      rootFile()->rewind();
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  bool RootPrimaryFileSequence::skipEvents(int offset) {
    assert(rootFile());
    while (offset != 0) {
      bool atEnd = rootFile()->skipEvents(offset);
      if ((offset > 0 || atEnd) && !nextFile()) {
        return false;
      }
      if (offset < 0 && !previousFile()) {
        setNoMoreFiles();
        return false;
      }
    }
    return true;
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
      // Save the current file and position so that we can restore them
      // if we fail to restore the desired event
      bool closedOriginalFile = false;
      size_t const originalFileSequenceNumber = sequenceNumberOfFile();
      IndexIntoFile::IndexIntoFileItr originalPosition = rootFile()->indexIntoFileIter();

      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      for (auto it = indexesIntoFiles().begin(), itEnd = indexesIntoFiles().end(); it != itEnd; ++it) {
        if (*it && (*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
          // We found it. Close the currently open file, and open the correct one.
          setAtFileSequenceNumber(it - indexesIntoFiles().begin());
          initFile(false);
          // Now get the item from the correct file.
          assert(rootFile());
          bool found = rootFile()->goToEvent(eventID);
          assert(found);
          return true;
        }
      }
      // Look for item in files not yet opened.
      for (auto it = indexesIntoFiles().begin(), itEnd = indexesIntoFiles().end(); it != itEnd; ++it) {
        if (!*it) {
          setAtFileSequenceNumber(it - indexesIntoFiles().begin());
          initFile(false);
          closedOriginalFile = true;
          if ((*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
            assert(rootFile());
            if (rootFile()->goToEvent(eventID)) {
              return true;
            }
          }
        }
      }
      if (closedOriginalFile) {
        setAtFileSequenceNumber(originalFileSequenceNumber);
        initFile(false);
        assert(rootFile());
        rootFile()->setPosition(originalPosition);
      }
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

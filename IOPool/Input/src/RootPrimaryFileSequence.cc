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
  RootPrimaryFileSequence::RootPrimaryFileSequence(
                ParameterSet const& pset,
                PoolSource& input,
                InputFileCatalog const& catalog,
                unsigned int nStreams) :
    RootInputFileSequence(pset, catalog),
    input_(input),
    firstFile_(true),
    branchesMustMatch_(BranchDescription::Permissive),
    orderedProcessHistoryIDs_(),
    nStreams_(nStreams),
    eventSkipperByID_(EventSkipperByID::create(pset).release()),
    // The default value provided as the second argument to the getUntrackedParameter function call
    // is not used when the ParameterSet has been validated and the parameters are not optional
    // in the description.  This is currently true when PoolSource is the primary input source.
    // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
    // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
    // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
    // and should be deleted from the code.
    initialNumberOfEventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0U)),
    noEventSort_(pset.getUntrackedParameter<bool>("noEventSort", true)),
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
    bypassVersionCheck_(pset.getUntrackedParameter<bool>("bypassVersionCheck", false)),
    treeCacheSize_(noEventSort_ ? pset.getUntrackedParameter<unsigned int>("cacheSize", roottree::defaultCacheSize) : 0U),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber", 0U)),
    productSelectorRules_(pset, "inputCommands", "InputSource"),
    duplicateChecker_(new DuplicateChecker(pset)),
    dropDescendants_(pset.getUntrackedParameter<bool>("dropDescendantsOfDroppedBranches", true)),
    labelRawDataLikeMC_(pset.getUntrackedParameter<bool>("labelRawDataLikeMC", true)),
    usingGoToEvent_(false),
    enablePrefetching_(false) {

    // The SiteLocalConfig controls the TTreeCache size and the prefetching settings.
    Service<SiteLocalConfig> pSLC;
    if(pSLC.isAvailable()) {
      if(treeCacheSize_ != 0U && pSLC->sourceTTreeCacheSize()) {
        treeCacheSize_ = *(pSLC->sourceTTreeCacheSize());
      }
      enablePrefetching_ = pSLC->enablePrefetching();
    }

    std::string branchesMustMatch = pset.getUntrackedParameter<std::string>("branchesMustMatch", std::string("permissive"));
    if(branchesMustMatch == std::string("strict")) branchesMustMatch_ = BranchDescription::Strict;

    // Prestage the files
    for (setAtFirstFile(); !noMoreFiles(); setAtNextFile()) {
      StorageFactory::get()->stagein(fileName());
    }
    // Open the first file.
    for (setAtFirstFile(); !noMoreFiles(); setAtNextFile()) {
      initFile(skipBadFiles_);
      if(rootFile()) break;
    }
    if(rootFile()) {
      input_.productRegistryUpdate().updateFromInput(rootFile()->productRegistry()->productList());
      if(initialNumberOfEventsToSkip_ != 0) {
        skipEvents(initialNumberOfEventsToSkip_);
      }
    }
  }

  RootPrimaryFileSequence::~RootPrimaryFileSequence() {
  }

  void
  RootPrimaryFileSequence::endJob() {
    closeFile_();
  }

  std::unique_ptr<FileBlock>
  RootPrimaryFileSequence::readFile_() {
    if(firstFile_) {
      // The first input file has already been opened.
      firstFile_ = false;
      if(!rootFile()) {
        initFile(skipBadFiles_);
      }
    } else {
      if(!nextFile()) {
        assert(0);
      }
    }
    if(!rootFile()) {
      return std::unique_ptr<FileBlock>(new FileBlock);
    }
    return rootFile()->createFileBlock();
  }

  void
  RootPrimaryFileSequence::closeFile_() {
    // close the currently open file, if any, and delete the RootFile object.
    if(rootFile()) {
      std::unique_ptr<InputSource::FileCloseSentry>
      sentry(new InputSource::FileCloseSentry(input_, lfn(), usedFallback()));
      rootFile()->close();
      if(duplicateChecker_) duplicateChecker_->inputFileClosed();
      rootFile().reset();
    }
  }

  void
  RootPrimaryFileSequence::initFile_(bool skipBadFiles) {
    // If we are not duplicate checking across files and we are not using random access to find events,
    // then we can delete the IndexIntoFile for the file we are closing.
    // If we can't delete all of it, then we can delete the parts we do not need.
    bool deleteIndexIntoFile = !usingGoToEvent_ && !(duplicateChecker_ && duplicateChecker_->checkingAllFiles() && !duplicateChecker_->checkDisabled());
    initTheFile(skipBadFiles, deleteIndexIntoFile, &input_, "primaryFiles", InputType::Primary);
  }

  RootPrimaryFileSequence::RootFileSharedPtr
  RootPrimaryFileSequence::makeRootFile(std::shared_ptr<InputFile> filePtr) {
      size_t currentIndexIntoFile = sequenceNumberOfFile();
      return std::make_shared<RootFile>(
          fileName(),
          input_.processConfiguration(),
          logicalFileName(),
          filePtr,
          eventSkipperByID_,
          initialNumberOfEventsToSkip_ != 0,
          remainingEvents(),
          remainingLuminosityBlocks(),
	  nStreams_,
          treeCacheSize_,
          treeMaxVirtualSize_,
          input_.processingMode(),
          setRun_,
          noEventSort_,
          productSelectorRules_,
          InputType::Primary,
          input_.branchIDListHelper(),
          input_.thinnedAssociationsHelper(),
          std::vector<BranchID>(), // associationsFromSecondary_
          duplicateChecker_,
          dropDescendants_,
          input_.processHistoryRegistryForUpdate(),
          indexesIntoFiles(),
          currentIndexIntoFile,
          orderedProcessHistoryIDs_,
          bypassVersionCheck_,
          labelRawDataLikeMC_,
          usingGoToEvent_,
          enablePrefetching_);
  }

  bool RootPrimaryFileSequence::nextFile() {
    if(!noMoreFiles()) setAtNextFile();
    if(noMoreFiles()) {
      return false;
    }

    initFile(skipBadFiles_);

    if(rootFile()) {
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = input_.productRegistryUpdate().merge(*rootFile()->productRegistry(),
                                                                   fileName(),
                                                                   branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw Exception(errors::MismatchedInputFiles,"RootPrimaryFileSequence::nextFile()") << mergeInfo;
      }
    }
    return true;
  }

  bool RootPrimaryFileSequence::previousFile() {
    if(atFirstFile()) {
      return false;
    }
    setAtPreviousFile();;

    initFile(false);

    if(rootFile()) {
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = input_.productRegistryUpdate().merge(*rootFile()->productRegistry(),
                                                                   fileName(),
                                                                   branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw Exception(errors::MismatchedInputFiles,"RootPrimaryFileSequence::previousEvent()") << mergeInfo;
      }
    }
    if(rootFile()) rootFile()->setToLastEntry();
    return true;
  }

  InputSource::ItemType
  RootPrimaryFileSequence::getNextItemType(RunNumber_t& run, LuminosityBlockNumber_t& lumi, EventNumber_t& event) {
    if(noMoreFiles()) {
      return InputSource::IsStop;
    }
    if(firstFile_) {
      return InputSource::IsFile;
    }
    if(rootFile()) {
      IndexIntoFile::EntryType entryType = rootFile()->getNextItemType(run, lumi, event);
      if(entryType == IndexIntoFile::kEvent) {
        return InputSource::IsEvent;
      } else if(entryType == IndexIntoFile::kLumi) {
        return InputSource::IsLumi;
      } else if(entryType == IndexIntoFile::kRun) {
        return InputSource::IsRun;
      }
      assert(entryType == IndexIntoFile::kEnd);
    }
    if(atLastFile()) {
      return InputSource::IsStop;
    }
    return InputSource::IsFile;
  }

  // Rewind to before the first event that was read.
  void
  RootPrimaryFileSequence::rewind_() {
    if(!atFirstFile()) {
      closeFile_();
      setAtFirstFile();
    }
    if(!rootFile()) {
      initFile(false);
    }
    rewindFile();
    firstFile_ = true;
    if(rootFile()) {
      if(initialNumberOfEventsToSkip_ != 0) {
        skipEvents(initialNumberOfEventsToSkip_);
      }
    }
  }

  // Rewind to the beginning of the current file
  void
  RootPrimaryFileSequence::rewindFile() {
    if(rootFile()) rootFile()->rewind();
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  bool
  RootPrimaryFileSequence::skipEvents(int offset) {
    assert(rootFile());
    while(offset != 0) {
      bool atEnd = rootFile()->skipEvents(offset);
      if((offset > 0 || atEnd) && !nextFile()) {
        return false;
      }
      if(offset < 0 && !previousFile()) {
        setNoMoreFiles();
        return false;
      }
    }
    return true;
  }

  bool
  RootPrimaryFileSequence::goToEvent(EventID const& eventID) {
    usingGoToEvent_ = true;
    if(rootFile()) {
      if(rootFile()->goToEvent(eventID)) {
        return true;
      }
      // If only one input file, give up now, to save time.
      if(rootFile() && indexesIntoFiles().size() == 1) {
        return false;
      }
      // Save the current file and position so that we can restore them
      // if we fail to restore the desired event
      bool closedOriginalFile = false;
      size_t const originalFileSequenceNumber = sequenceNumberOfFile();
      IndexIntoFile::IndexIntoFileItr originalPosition = rootFile()->indexIntoFileIter();

      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      typedef std::vector<std::shared_ptr<IndexIntoFile> >::const_iterator Iter;
      for(Iter it = indexesIntoFiles().begin(), itEnd = indexesIntoFiles().end(); it != itEnd; ++it) {
        if(*it && (*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
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
      for(Iter it = indexesIntoFiles().begin(), itEnd = indexesIntoFiles().end(); it != itEnd; ++it) {
        if(!*it) {
          setAtFileSequenceNumber(it - indexesIntoFiles().begin());
          initFile(false);
          closedOriginalFile = true;
          if((*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
            assert(rootFile());
            if(rootFile()->goToEvent(eventID)) {
              return true;
            }
          }
        }
      }
      if(closedOriginalFile) {
        setAtFileSequenceNumber(originalFileSequenceNumber);
        initFile(false);
        assert(rootFile());
        rootFile()->setPosition(originalPosition);
      }
    }
    return false;
  }

  int
  RootPrimaryFileSequence::remainingEvents() const {
    return input_.remainingEvents();
  }

  int
  RootPrimaryFileSequence::remainingLuminosityBlocks() const {
    return input_.remainingLuminosityBlocks();
  }

  void
  RootPrimaryFileSequence::fillDescription(ParameterSetDescription & desc) {
    desc.addUntracked<unsigned int>("skipEvents", 0U)
        ->setComment("Skip the first 'skipEvents' events that otherwise would have been processed.");
    desc.addUntracked<bool>("noEventSort", true)
        ->setComment("True:  Process runs, lumis and events in the order they appear in the file (but see notes 1 and 2).\n"
                     "False: Process runs, lumis and events in each file in numerical order (run#, lumi#, event#) (but see note 3).\n"
                     "Note 1: Events within the same lumi will always be processed contiguously.\n"
                     "Note 2: Lumis within the same run will always be processed contiguously.\n"
                     "Note 3: Any sorting occurs independently in each input file (no sorting across input files).");
    desc.addUntracked<bool>("skipBadFiles", false)
        ->setComment("True:  Ignore any missing or unopenable input file.\n"
                     "False: Throw exception if missing or unopenable input file.");
    desc.addUntracked<bool>("bypassVersionCheck", false)
        ->setComment("True:  Bypass release version check.\n"
                     "False: Throw exception if reading file in a release prior to the release in which the file was written.");
    desc.addUntracked<unsigned int>("cacheSize", roottree::defaultCacheSize)
        ->setComment("Size of ROOT TTree prefetch cache.  Affects performance.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache.  Affects performance.");
    desc.addUntracked<unsigned int>("setRunNumber", 0U)
        ->setComment("If non-zero, change number of first run to this number. Apply same offset to all runs.  Allowed only for simulation.");
    desc.addUntracked<bool>("dropDescendantsOfDroppedBranches", true)
        ->setComment("If True, also drop on input any descendent of any branch dropped on input.");
    std::string defaultString("permissive");
    desc.addUntracked<std::string>("branchesMustMatch", defaultString)
        ->setComment("'strict':     Branches in each input file must match those in the first file.\n"
                     "'permissive': Branches in each input file may be any subset of those in the first file.");
    desc.addUntracked<bool>("labelRawDataLikeMC", true)
        ->setComment("If True: replace module label for raw data to match MC. Also use 'LHC' as process.");

    ProductSelectorRules::fillDescription(desc, "inputCommands");
    EventSkipperByID::fillDescription(desc);
    DuplicateChecker::fillDescription(desc);
  }

  ProcessingController::ForwardState
  RootPrimaryFileSequence::forwardState() const {
    if(rootFile()) {
      if(!rootFile()->wasLastEventJustRead()) {
        return ProcessingController::kEventsAheadInFile;
      }
      if(noMoreFiles() || atLastFile()) { 
        return ProcessingController::kAtLastEvent;
      } else {
        return ProcessingController::kNextFileExists;
      }
    }
    return ProcessingController::kUnknownForward;
  }

  ProcessingController::ReverseState
  RootPrimaryFileSequence::reverseState() const {
    if(rootFile()) {
      if(!rootFile()->wasFirstEventJustRead()) {
        return ProcessingController::kEventsBackwardsInFile;
      }
      if(!atFirstFile()) {
        return ProcessingController::kPreviousFileExists;
      }
      return ProcessingController::kAtFirstEvent;
    }
    return ProcessingController::kUnknownReverse;
  }

}

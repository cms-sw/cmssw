/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "RootInputFileSequence.h"
#include "PoolSource.h"
#include "RootFile.h"
#include "RootTree.h"
#include "DuplicateChecker.h"

#include "FWCore/Catalog/interface/FileCatalog.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "DataFormats/Provenance/interface/FileIndex.h"

#include "CLHEP/Random/RandFlat.h"
#include "TFile.h"
#include "TSystem.h"

#include <ctime>

namespace edm {
  RootInputFileSequence::RootInputFileSequence(
		ParameterSet const& pset,
		PoolSource const& input,
		InputFileCatalog const& catalog,
		bool primarySequence) :
    input_(input),
    catalog_(catalog),
    firstFile_(true),
    fileIterBegin_(fileCatalogItems().begin()),
    fileIterEnd_(fileCatalogItems().end()),
    fileIter_(fileIterEnd_),
    rootFile_(),
    parametersMustMatch_(BranchDescription::Permissive),
    branchesMustMatch_(BranchDescription::Permissive),
    flatDistribution_(),
    fileIndexes_(fileCatalogItems().size()),
    eventsRemainingInFile_(0),
    startAtRun_(pset.getUntrackedParameter<unsigned int>("firstRun", 1U)),
    startAtLumi_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1U)),
    startAtEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1U)),
    currentRun_(0U),
    currentLumi_(0U),
    skippedToRun_(0U),
    skippedToLumi_(0U),
    skippedToEvent_(0U),
    skippedToEntry_(FileIndex::Element::invalidEntry),
    skipEvents_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0U)),
    whichLumisToSkip_(pset.getUntrackedParameter<std::vector<LuminosityBlockRange> >("lumisToSkip", std::vector<LuminosityBlockRange>())),
    whichLumisToProcess_(pset.getUntrackedParameter<std::vector<LuminosityBlockRange> >("lumisToProcess", std::vector<LuminosityBlockRange>())),
    whichEventsToSkip_(pset.getUntrackedParameter<std::vector<EventRange> >("eventsToSkip",std::vector<EventRange>())),
    whichEventsToProcess_(pset.getUntrackedParameter<std::vector<EventRange> >("eventsToProcess",std::vector<EventRange>())),
    noEventSort_(pset.getUntrackedParameter<bool>("noEventSort", false)),
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
    treeCacheSize_(pset.getUntrackedParameter<unsigned int>("cacheSize", 0U)),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    forcedRunOffset_(0),
    setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber", 0U)),
    groupSelectorRules_(pset, "inputCommands", "InputSource"),
    primarySequence_(primarySequence),
    duplicateChecker_(),
    dropDescendants_(pset.getUntrackedParameter<bool>("dropDescendantsOfDroppedBranches", primary()))
  {
    if(!primarySequence_) noEventSort_ = false;
    if(!whichLumisToProcess_.empty() && !whichEventsToProcess_.empty()) {
      throw edm::Exception(errors::Configuration)
        << "Illegal configuration options passed to PoolSource\n"
        << "You cannot request both \"luminosityBlocksToProcess\" and \"eventsToProcess\".\n";
    }

    if(primarySequence_ && primary()) duplicateChecker_.reset(new DuplicateChecker(pset));

    StorageFactory *factory = StorageFactory::get();
    for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_)
      factory->stagein(fileIter_->fileName());

    std::string parametersMustMatch = pset.getUntrackedParameter<std::string>("parametersMustMatch", std::string("permissive"));
    if(parametersMustMatch == std::string("strict")) parametersMustMatch_ = BranchDescription::Strict;

    // "fileMatchMode" is for backward compatibility.
    parametersMustMatch = pset.getUntrackedParameter<std::string>("fileMatchMode", std::string("permissive"));
    if(parametersMustMatch == std::string("strict")) parametersMustMatch_ = BranchDescription::Strict;

    std::string branchesMustMatch = pset.getUntrackedParameter<std::string>("branchesMustMatch", std::string("permissive"));
    if(branchesMustMatch == std::string("strict")) branchesMustMatch_ = BranchDescription::Strict;

    if(primary()) {
      for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_) {
        initFile(skipBadFiles_);
        if(rootFile_) break;
      }
      if(rootFile_) {
        forcedRunOffset_ = rootFile_->setForcedRunOffset(setRun_);
        if(forcedRunOffset_ < 0) {
          throw edm::Exception(errors::Configuration)
            << "The value of the 'setRunNumber' parameter must not be\n"
            << "less than the first run number in the first input file.\n"
            << "'setRunNumber' was " << setRun_ <<", while the first run was "
            << setRun_ - forcedRunOffset_ << ".\n";
        }
        productRegistryUpdate().updateFromInput(rootFile_->productRegistry()->productList());
	if(primarySequence_) {
          BranchIDListHelper::updateFromInput(rootFile_->branchIDLists(), fileIter_->fileName());
	  if(skipEvents_ != 0) {
	    skipEvents(skipEvents_);
	  }
	}
      }
    }
  }

  std::vector<FileCatalogItem> const&
  RootInputFileSequence::fileCatalogItems() const {
    return catalog_.fileCatalogItems();
  }

  void
  RootInputFileSequence::endJob() {
    if(primary()) {
      closeFile_();
    } else {
      if(rootFile_) {
        rootFile_->close(true);
        logFileAction("  Closed file ", rootFile_->file());
        rootFile_.reset();
      }
    }
  }

  boost::shared_ptr<FileBlock>
  RootInputFileSequence::readFile_() {
    if(firstFile_) {
      // The first input file has already been opened.
      firstFile_ = false;
      if(!rootFile_) {
	initFile(skipBadFiles_);
      }
    } else {
      if(!nextFile()) {
        assert(0);
      }
    }
    if(!rootFile_) {
      return boost::shared_ptr<FileBlock>(new FileBlock);
    }
    return rootFile_->createFileBlock();
  }

  void RootInputFileSequence::closeFile_() {
    if(rootFile_) {
    // Account for events skipped in the file.
      {
        std::auto_ptr<InputSource::FileCloseSentry> 
	  sentry((primarySequence_ && primary()) ? new InputSource::FileCloseSentry(input_) : 0);
        rootFile_->close(primary());
      }
      logFileAction("  Closed file ", rootFile_->file());
      // The next step is necessary for the duplicate checking to work properly
      if(noEventSort_) rootFile_->fileIndexSharedPtr()->sortBy_Run_Lumi_Event();
      rootFile_.reset();
      if(duplicateChecker_) duplicateChecker_->inputFileClosed();
    }
  }

  void RootInputFileSequence::initFile(bool skipBadFiles) {
    // close the currently open file, any, and delete the RootFile object.
    closeFile_();
    boost::shared_ptr<TFile> filePtr;
    try {
      logFileAction("  Initiating request to open file ", fileIter_->fileName());
      std::auto_ptr<InputSource::FileOpenSentry> 
	sentry((primarySequence_ && primary()) ? new InputSource::FileOpenSentry(input_) : 0);
      filePtr.reset(TFile::Open(gSystem->ExpandPathName(fileIter_->fileName().c_str())));
    }
    catch (cms::Exception e) {
      if(!skipBadFiles) {
	throw edm::Exception(edm::errors::FileOpenError) << e.explainSelf() << "\n" <<
	   "RootInputFileSequence::initFile(): Input file " << fileIter_->fileName() << " was not found or could not be opened.\n";
      }
    }
    if(filePtr && !filePtr->IsZombie()) {
      logFileAction("  Successfully opened file ", fileIter_->fileName());
      std::vector<boost::shared_ptr<FileIndex> >::size_type currentFileIndex = fileIter_ - fileIterBegin_;
      rootFile_ = RootFileSharedPtr(new RootFile(fileIter_->fileName(), catalog_.url(),
	  processConfiguration(), fileIter_->logicalFileName(), filePtr,
	  startAtRun_, startAtLumi_, startAtEvent_, skipEvents_ != 0,
	  whichLumisToSkip_, whichEventsToSkip_,
	  remainingEvents(), remainingLuminosityBlocks(), treeCacheSize_, treeMaxVirtualSize_,
	  input_.processingMode(),
	  forcedRunOffset_,
	  whichLumisToProcess_, whichEventsToProcess_,
	  noEventSort_,
	  groupSelectorRules_, !primarySequence_, duplicateChecker_, dropDescendants_,
          fileIndexes_, currentFileIndex));
          fileIndexes_[currentFileIndex] = rootFile_->fileIndexSharedPtr();
      rootFile_->reportOpened(primary() ?
	 (primarySequence_ ? "primaryFiles" : "secondaryFiles") : "mixingFiles");
    } else {
      if(!skipBadFiles) {
	throw edm::Exception(edm::errors::FileOpenError) <<
	   "RootInputFileSequence::initFile(): Input file " << fileIter_->fileName() << " was not found or could not be opened.\n";
      }
      LogWarning("") << "Input file: " << fileIter_->fileName() << " was not found or could not be opened, and will be skipped.\n";
    }
  }

  ProductRegistry const&
  RootInputFileSequence::fileProductRegistry() const {
    return *rootFile_->productRegistry();
  }

  bool RootInputFileSequence::nextFile() {
    if(fileIter_ != fileIterEnd_) ++fileIter_;
    if(fileIter_ == fileIterEnd_) {
      if(primarySequence_) {
	return false;
      } else {
	fileIter_ = fileIterBegin_;
      }
    }

    initFile(skipBadFiles_);

    if(primarySequence_ && rootFile_) {
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    parametersMustMatch_,
							    branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw edm::Exception(errors::MismatchedInputFiles,"RootInputFileSequence::nextFile()") << mergeInfo;
      }
      BranchIDListHelper::updateFromInput(rootFile_->branchIDLists(), fileIter_->fileName());
    }
    return true;
  }

  bool RootInputFileSequence::previousFile() {
    if(fileIter_ == fileIterBegin_) {
      if(primarySequence_) {
	return false;
      } else {
	fileIter_ = fileIterEnd_;
      }
    }
    --fileIter_;

    initFile(false);

    if(primarySequence_ && rootFile_) {
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    parametersMustMatch_,
							    branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw edm::Exception(errors::MismatchedInputFiles,"RootInputFileSequence::previousEvent()") << mergeInfo;
      }
      BranchIDListHelper::updateFromInput(rootFile_->branchIDLists(), fileIter_->fileName());
    }
    if(rootFile_) rootFile_->setToLastEntry();
    return true;
  }

  RootInputFileSequence::~RootInputFileSequence() {
  }

  boost::shared_ptr<RunPrincipal>
  RootInputFileSequence::readRun_() {
    boost::shared_ptr<RunPrincipal> rp = rootFile_->readRun(primarySequence_ ? productRegistry() : rootFile_->productRegistry()); 
    currentRun_ = (rp ? rp->run() : 0U);
    currentLumi_ = 0U;
    return rp;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootInputFileSequence::readLuminosityBlock_() {
    boost::shared_ptr<LuminosityBlockPrincipal> lbp = rootFile_->readLumi(primarySequence_ ? productRegistry() : rootFile_->productRegistry(), runPrincipal()); 
    currentLumi_ = (lbp ? lbp->luminosityBlock() : 0U);
    return lbp;
  }

  // readEvent_() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  //   1. create an EventPrincipal with a unique EventID
  //   2. For each entry in the provenance, put in one Group,
  //      holding the Provenance for the corresponding EDProduct.
  //   3. set up the caches in the EventPrincipal to know about this
  //      Group.
  //
  // We do *not* create the EDProduct instance (the equivalent of reading
  // the branch containing this EDProduct. That will be done by the Delayed Reader,
  //  when it is asked to do so.
  //

  std::auto_ptr<EventPrincipal>
  RootInputFileSequence::readEvent_() {
    return rootFile_->readEvent(primarySequence_ ? productRegistry() : rootFile_->productRegistry()); 
  }

  std::auto_ptr<EventPrincipal>
  RootInputFileSequence::readCurrentEvent() {
    return rootFile_->readCurrentEvent(primarySequence_ ?
				       productRegistry() :
				       rootFile_->productRegistry()); 
  }

  InputSource::ItemType
  RootInputFileSequence::getNextItemType() {
    if(fileIter_ == fileIterEnd_) {
      return InputSource::IsStop;
    }
    if(firstFile_) {
      return InputSource::IsFile;
    }
    if(rootFile_) {
      if(skippedToRun_) {
	rootFile_->setEntryAtRun(skippedToRun_);
	skippedToRun_ = 0U;
      } else if(skippedToLumi_) {
	rootFile_->setEntryAtLumi(rootFile_->runNumber(), skippedToLumi_);
	skippedToLumi_ = 0U;
      } else if(skippedToEvent_) {
	rootFile_->setEntryAtEventEntry(rootFile_->runNumber(), rootFile_->luminosityBlockNumber(), skippedToEvent_, skippedToEntry_, true);
	skippedToEvent_ = 0U;
	skippedToEntry_ = FileIndex::Element::invalidEntry;
      }
      FileIndex::EntryType entryType = rootFile_->getNextEntryTypeWanted();
      if(entryType == FileIndex::kEvent) {
        return InputSource::IsEvent;
      } else if(entryType == FileIndex::kLumi) {
        return InputSource::IsLumi;
      } else if(entryType == FileIndex::kRun) {
        return InputSource::IsRun;
      }
      assert(entryType == FileIndex::kEnd);
    }
    if(fileIter_ + 1 == fileIterEnd_) {
      return InputSource::IsStop;
    }
    return InputSource::IsFile;
  }

  // Rewind to before the first event that was read.
  void
  RootInputFileSequence::rewind_() {
    if (fileIter_ != fileIterBegin_) {
      closeFile_();
      fileIter_ = fileIterBegin_;
      initFile(false);
    }
    rewindFile();
  }

  // Rewind to the beginning of the current file
  void
  RootInputFileSequence::rewindFile() {
    rootFile_->rewind();
    currentRun_ = skippedToRun_ = 0U;
    currentLumi_ = skippedToLumi_ = 0U;
    skippedToEvent_ = 0U;
    skippedToEntry_ = FileIndex::Element::invalidEntry;
  }

  void
  RootInputFileSequence::reset() {
    //NOTE: Need to handle duplicate checker
    // Also what if skipBadFiles_==true and the first time we succeeded but after a reset we fail?
    if(primary()) {
      firstFile_=true;
      for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_) {
        initFile(skipBadFiles_);
        if(rootFile_) break;
      }
      if(rootFile_) {
        forcedRunOffset_ = rootFile_->setForcedRunOffset(setRun_);
        if(forcedRunOffset_ < 0) {
          throw edm::Exception(errors::Configuration)
          << "The value of the 'setRunNumber' parameter must not be\n"
          << "less than the first run number in the first input file.\n"
          << "'setRunNumber' was " << setRun_ <<", while the first run was "
          << setRun_ - forcedRunOffset_ << ".\n";
        }
	if(primarySequence_) {
          BranchIDListHelper::updateFromInput(rootFile_->branchIDLists(), fileIter_->fileName());
	  if(skipEvents_ != 0) {
	    skipEvents(skipEvents_);
	  }
	}
      }
    }    
  }
  
  // Advance "offset" events.  Offset can be positive or negative (or zero).
  bool
  RootInputFileSequence::skipEvents(int offset) {
    assert (skipEvents_ == 0 || skipEvents_ == offset);
    skipEvents_ = offset;
    while(skipEvents_ != 0) {
      skipEvents_ = rootFile_->skipEvents(skipEvents_);
      if(skipEvents_ > 0 && !nextFile()) {
	skipEvents_ = 0;
	return false;
      }
      if(skipEvents_ < 0 && !previousFile()) {
	skipEvents_ = 0;
        return false;
      }
    }
    rootFile_->skipEvents(0);
    setSkipInfo();
    return true;
  }

  void
  RootInputFileSequence::setSkipInfo() {
    if(rootFile_->runNumber() != currentRun_) {
      // Skipped to different run
      // Save info to provide run and lumi transitions.
      skippedToRun_ = rootFile_->runNumber();
      skippedToLumi_ = rootFile_->luminosityBlockNumber();
      skippedToEvent_ = rootFile_->eventNumber();
      skippedToEntry_ = rootFile_->entryNumber();
    } else if(rootFile_->luminosityBlockNumber() != currentLumi_) {
      // Skipped to different lumi in same run
      // Save info to provide lumi transition.
      skippedToRun_ = 0U;
      skippedToLumi_ = rootFile_->luminosityBlockNumber();
      skippedToEvent_ = rootFile_->eventNumber();
      skippedToEntry_ = rootFile_->entryNumber();
    } else {
      // Skipped to event in same lumi.  No need to save anything. 
      skippedToRun_ = 0U;
      skippedToLumi_ = 0U;
      skippedToEvent_ = 0U;
      skippedToEntry_ = FileIndex::Element::invalidEntry;
    }
  }
  

  bool
  RootInputFileSequence::skipToItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact, bool record) {
    // Note: 'exact' argumet is ignored unless the item is an event.
    // Attempt to find item in currently open input file.
    bool found = rootFile_->setEntryAtItem(run, lumi, event, exact);
    if(!found) {
      // If only one input file, give up now, to save time.
      if(fileIndexes_.size() == 1) {
	return false;
      }
      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      typedef std::vector<boost::shared_ptr<FileIndex> >::const_iterator Iter;
      for(Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if(*it && (*it)->containsItem(run, lumi, event, exact)) {
          // We found it. Close the currently open file, and open the correct one.
	  fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
	  // Now get the item from the correct file.
          found = rootFile_->setEntryAtItem(run, lumi, event, exact);
	  assert (found);
	  if (record) setSkipInfo();	  
	  return true;
	}
      }
      // Look for item in files not yet opened.
      for(Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if(!*it) {
	  fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
          found = rootFile_->setEntryAtItem(run, lumi, event, exact);
	  if(found) {
	    if (record) setSkipInfo();	  
	    return true;
	  }
	}
      }
      // Not found
      return false;
    }
    if (record) setSkipInfo();	  
    return true;
  }

  bool const
  RootInputFileSequence::primary() const {
    return input_.primary();
  }

  boost::shared_ptr<RunPrincipal>
  RootInputFileSequence::runPrincipal() const {
    return input_.runPrincipal();
  }
   
  ProcessConfiguration const&
  RootInputFileSequence::processConfiguration() const {
    return input_.processConfiguration();
  }
  
  int
  RootInputFileSequence::remainingEvents() const {
    return input_.remainingEvents();
  }

  int
  RootInputFileSequence::remainingLuminosityBlocks() const {
    return input_.remainingLuminosityBlocks();
  }

  ProductRegistry &
  RootInputFileSequence::productRegistryUpdate() const{
    return input_.productRegistryUpdate();
  }

  boost::shared_ptr<ProductRegistry const>
  RootInputFileSequence::productRegistry() const{
    return input_.productRegistry();
  }

  void
  RootInputFileSequence::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    std::vector<std::string> rules;
    rules.reserve(wantedBranches.size() + 1);
    rules.push_back(std::string("drop *")); 
    for(std::vector<std::string>::const_iterator it = wantedBranches.begin(), itEnd = wantedBranches.end();
	it != itEnd; ++it) {
      rules.push_back("keep " + *it + "_*");
    }
    ParameterSet pset;
    pset.addUntrackedParameter("inputCommands", rules);
    groupSelectorRules_ = GroupSelectorRules(pset, "inputCommands", "InputSource");
  }

  void
  RootInputFileSequence::readMany_(int number, EventPrincipalVector& result) {
    for(int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if(ev.get() == 0) {
	return;
      }
      VectorInputSource::EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readMany_(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber) {
    unsigned int currentSeqNumber = fileIter_ - fileIterBegin_;
    if(currentSeqNumber != fileSeqNumber) {
      fileIter_ = fileIterBegin_ + fileSeqNumber;
      initFile(false);
    }
    rootFile_->setEntryAtEvent(id.run(), 0U, id.event(), false);
    for(int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if(ev.get() == 0) {
        rewindFile();
	ev = readCurrentEvent();
	assert(ev.get() != 0);
      }
      VectorInputSource::EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    if (!flatDistribution_) {
      Service<RandomNumberGenerator> rng;
      CLHEP::HepRandomEngine& engine = rng->getEngine();
      flatDistribution_.reset(new CLHEP::RandFlat(engine));
    }
    skipBadFiles_ = false;
    unsigned int currentSeqNumber = fileIter_ - fileIterBegin_;
    while(eventsRemainingInFile_ < number) {
      fileIter_ = fileIterBegin_ + flatDistribution_->fireInt(fileCatalogItems().size());
      unsigned int newSeqNumber = fileIter_ - fileIterBegin_;
      if(newSeqNumber != currentSeqNumber) {
        initFile(false);
      }
      eventsRemainingInFile_ = rootFile_->eventTree().entries();
      if(eventsRemainingInFile_ == 0) {
	throw edm::Exception(edm::errors::NotFound) <<
	   "RootInputFileSequence::readManyRandom_(): Secondary Input file " << fileIter_->fileName() << " contains no events.\n";
      }
      rootFile_->setAtEventEntry(flatDistribution_->fireInt(eventsRemainingInFile_));
    }
    fileSeqNumber = fileIter_ - fileIterBegin_;
    for(int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if(ev.get() == 0) {
        rewindFile();
	ev = readCurrentEvent();
	assert(ev.get() != 0);
      }
       VectorInputSource::EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
      --eventsRemainingInFile_;
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    skipBadFiles_ = false;
    if (fileIter_ == fileIterEnd_ || !rootFile_) {
      fileIter_ = fileIterBegin_;
      initFile(false);
      rootFile_->setAtEventEntry(0);
    }
    fileSeqNumber = fileIter_ - fileIterBegin_;
    unsigned int numberRead = 0;
    for(int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if(ev.get() == 0) {
	if (numberRead == 0) {
	  ++fileIter_;
          fileSeqNumber = fileIter_ - fileIterBegin_;
	  if (fileIter_ == fileIterEnd_) {
	    return;
	  }
	  initFile(false);
	  rootFile_->setAtEventEntry(0);
	  return readManySequential_(number, result, fileSeqNumber);
	}
	return;
      }
      VectorInputSource::EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
      ++numberRead;
      rootFile_->nextEventEntry();
    }
  }

  void RootInputFileSequence::logFileAction(const char* msg, std::string const& file) {
    if(primarySequence_) {
      time_t t = time(0);
      char ts[] = "dd-Mon-yyyy hh:mm:ss TZN     ";
      strftime( ts, strlen(ts)+1, "%d-%b-%Y %H:%M:%S %Z", localtime(&t) );
      LogAbsolute("fileAction") << ts << msg << file;
      FlushMessageLog();
    }
  }
}


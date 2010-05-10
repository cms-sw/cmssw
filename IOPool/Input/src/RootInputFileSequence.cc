/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "RootInputFileSequence.h"
#include "PoolSource.h"
#include "RootFile.h"
#include "RootTree.h"
#include "DuplicateChecker.h"

#include "FWCore/Catalog/interface/FileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/src/PrincipalCache.h"
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
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CLHEP/Random/RandFlat.h"
#include "TFile.h"
#include "TSystem.h"

#include <ctime>

namespace edm {
  RootInputFileSequence::RootInputFileSequence(
		ParameterSet const& pset,
		PoolSource const& input,
		InputFileCatalog const& catalog,
		PrincipalCache& cache,
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
    eventSkipperByID_(primarySequence ? EventSkipperByID::create(pset).release() : 0),
    eventsRemainingInFile_(0),
    currentRun_(0U),
    currentLumi_(0U),
    skippedToRun_(0U),
    skippedToLumi_(0U),
    skippedToEvent_(0U),
    skippedToEntry_(FileIndex::Element::invalidEntry),
    // The default value provided as the second argument to the getUntrackedParameter function call
    // is not used when the ParameterSet has been validated and the parameters are not optional
    // in the description.  This is currently true when PoolSource is the primary input source.
    // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
    // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
    // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
    // and should be deleted from the code.
    numberOfEventsToSkip_(primarySequence ? pset.getUntrackedParameter<unsigned int>("skipEvents", 0U) : 0U),
    noEventSort_(primarySequence ? pset.getUntrackedParameter<bool>("noEventSort", false) : false),
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
    treeCacheSize_(pset.getUntrackedParameter<unsigned int>("cacheSize", input::defaultCacheSize)),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber", 0U)),
    groupSelectorRules_(pset, "inputCommands", "InputSource"),
    primarySequence_(primarySequence),
    duplicateChecker_(primarySequence ? new DuplicateChecker(pset) : 0),
    dropDescendants_(pset.getUntrackedParameter<bool>("dropDescendantsOfDroppedBranches", primary())) {

    //we now allow the site local config to specify what the TTree cache size should be
    edm::Service<edm::SiteLocalConfig> pSLC;
    if(pSLC.isAvailable() && pSLC->sourceTTreeCacheSize()) {
      treeCacheSize_=*(pSLC->sourceTTreeCacheSize());
    }
    StorageFactory *factory = StorageFactory::get();
    for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_)
      factory->stagein(fileIter_->fileName());

    std::string parametersMustMatch = pset.getUntrackedParameter<std::string>("parametersMustMatch", std::string("permissive"));
    if(parametersMustMatch == std::string("strict")) parametersMustMatch_ = BranchDescription::Strict;

    std::string branchesMustMatch = pset.getUntrackedParameter<std::string>("branchesMustMatch", std::string("permissive"));
    if(branchesMustMatch == std::string("strict")) branchesMustMatch_ = BranchDescription::Strict;

    if(primary()) {
      for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_) {
        initFile(skipBadFiles_);
        if(rootFile_) break;
      }
      if(rootFile_) {
        productRegistryUpdate().updateFromInput(rootFile_->productRegistry()->productList());
	if(numberOfEventsToSkip_ != 0) {
	  skipEvents(numberOfEventsToSkip_, cache);
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
  RootInputFileSequence::readFile_(PrincipalCache& cache) {
    if(firstFile_) {
      // The first input file has already been opened.
      firstFile_ = false;
      if(!rootFile_) {
	initFile(skipBadFiles_);
      }
    } else {
      if(!nextFile(cache)) {
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
	  eventSkipperByID_, numberOfEventsToSkip_ != 0,
	  remainingEvents(), remainingLuminosityBlocks(), treeCacheSize_, treeMaxVirtualSize_,
	  input_.processingMode(),
	  setRun_,
	  noEventSort_,
	  groupSelectorRules_, !primarySequence_, duplicateChecker_, dropDescendants_,
          fileIndexes_, currentFileIndex));
          fileIndexes_[currentFileIndex] = rootFile_->fileIndexSharedPtr();
      rootFile_->reportOpened(primary() ?
	 (primarySequence_ ? "primaryFiles" : "secondaryFiles") : "mixingFiles");
      if (primarySequence_) {
        BranchIDListHelper::updateFromInput(rootFile_->branchIDLists(), fileIter_->fileName());
      }
    } else {
      if(!skipBadFiles) {
	throw edm::Exception(edm::errors::FileOpenError) <<
	   "RootInputFileSequence::initFile(): Input file " << fileIter_->fileName() << " was not found or could not be opened.\n";
      }
      LogWarning("") << "Input file: " << fileIter_->fileName() << " was not found or could not be opened, and will be skipped.\n";
    }
  }

  boost::shared_ptr<ProductRegistry const>
  RootInputFileSequence::fileProductRegistry() const {
    return rootFile_->productRegistry();
  }

  bool RootInputFileSequence::nextFile(PrincipalCache& cache) {
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
      size_t size = productRegistry()->size();
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    parametersMustMatch_,
							    branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw edm::Exception(errors::MismatchedInputFiles,"RootInputFileSequence::nextFile()") << mergeInfo;
      }
      if (productRegistry()->size() > size) {
        cache.adjustIndexesAfterProductRegistryAddition();
      }
      cache.adjustEventToNewProductRegistry(productRegistry());
    }
    return true;
  }

  bool RootInputFileSequence::previousFile(PrincipalCache& cache) {
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
      size_t size = productRegistry()->size();
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    parametersMustMatch_,
							    branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw edm::Exception(errors::MismatchedInputFiles,"RootInputFileSequence::previousEvent()") << mergeInfo;
      }
      if (productRegistry()->size() > size) {
        cache.adjustIndexesAfterProductRegistryAddition();
      }
      cache.adjustEventToNewProductRegistry(productRegistry());
    }
    if(rootFile_) rootFile_->setToLastEntry();
    return true;
  }

  RootInputFileSequence::~RootInputFileSequence() {
  }

  boost::shared_ptr<RunAuxiliary>
  RootInputFileSequence::readRunAuxiliary_() {
    boost::shared_ptr<RunAuxiliary> aux = rootFile_->readRunAuxiliary_();
    currentRun_ = (aux ? aux->run() : 0U);
    currentLumi_ = 0U;
    return aux;
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  RootInputFileSequence::readLuminosityBlockAuxiliary_() {
    boost::shared_ptr<LuminosityBlockAuxiliary> aux = rootFile_->readLuminosityBlockAuxiliary_();
    currentLumi_ = (aux ? aux->luminosityBlock() : 0U);
    return aux;
  }

  boost::shared_ptr<RunPrincipal>
  RootInputFileSequence::readRun_(boost::shared_ptr<RunPrincipal> rpCache) {
    return rootFile_->readRun_(rpCache);
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootInputFileSequence::readLuminosityBlock_(boost::shared_ptr<LuminosityBlockPrincipal> lbCache) {
    return rootFile_->readLumi(lbCache);
  }

  // readEvent() is responsible for setting up the EventPrincipal.
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

  EventPrincipal*
  RootInputFileSequence::readEvent(EventPrincipal& cache, boost::shared_ptr<LuminosityBlockPrincipal> lb) {
    return rootFile_->readEvent(cache, lb);
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
    }
    if (!rootFile_) {
      initFile(false);
    }
    rewindFile();
    firstFile_ = true;
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
  RootInputFileSequence::reset(PrincipalCache& cache) {
    //NOTE: Need to handle duplicate checker
    // Also what if skipBadFiles_==true and the first time we succeeded but after a reset we fail?
    if(primary()) {
      firstFile_ = true;
      for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_) {
        initFile(skipBadFiles_);
        if(rootFile_) break;
      }
      if(rootFile_) {
	if(numberOfEventsToSkip_ != 0) {
	  skipEvents(numberOfEventsToSkip_, cache);
	}
      }
    }
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  bool
  RootInputFileSequence::skipEvents(int offset, PrincipalCache& cache) {
    assert (numberOfEventsToSkip_ == 0 || numberOfEventsToSkip_ == offset);
    numberOfEventsToSkip_ = offset;
    while(numberOfEventsToSkip_ != 0) {
      bool atEnd = rootFile_->skipEvents(numberOfEventsToSkip_);
      if((numberOfEventsToSkip_ > 0 || atEnd) && !nextFile(cache)) {
	numberOfEventsToSkip_ = 0;
	return false;
      }
      if(numberOfEventsToSkip_ < 0 && !previousFile(cache)) {
	numberOfEventsToSkip_ = 0;
        return false;
      }
    }
    int dummy = 0;
    bool atTheEnd = rootFile_->skipEvents(dummy);
    if (atTheEnd && !nextFile(cache)) {
      numberOfEventsToSkip_ = 0;
      return false;
    }
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
    bool found = rootFile_ && rootFile_->setEntryAtItem(run, lumi, event, exact);
    if(!found) {
      // If only one input file, give up now, to save time.
      if(rootFile_ && fileIndexes_.size() == 1) {
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
  RootInputFileSequence::readMany(int number, EventPrincipalVector& result) {
    for(int i = 0; i < number; ++i) {
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(rootFile_->productRegistry(), processConfiguration()));
      EventPrincipal* ev = rootFile_->readCurrentEvent(*ep);
      if(ev == 0) {
	return;
      }
      assert(ev == ep.get());
      result.push_back(ep);
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readMany(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber) {
    unsigned int currentSeqNumber = fileIter_ - fileIterBegin_;
    if(currentSeqNumber != fileSeqNumber) {
      fileIter_ = fileIterBegin_ + fileSeqNumber;
      initFile(false);
    }
    rootFile_->setEntryAtEvent(id.run(), 0U, id.event(), false);
    for(int i = 0; i < number; ++i) {
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(rootFile_->productRegistry(), processConfiguration()));
      EventPrincipal* ev = rootFile_->readCurrentEvent(*ep);
      if(ev == 0) {
        rewindFile();
	ev = rootFile_->readCurrentEvent(*ep);
      }
      assert(ev == ep.get());
      result.push_back(ep);
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readManyRandom(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    result.reserve(number);
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
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(rootFile_->productRegistry(), processConfiguration()));
      EventPrincipal* ev = rootFile_->readCurrentEvent(*ep);
      if(ev == 0) {
        rewindFile();
	ev = rootFile_->readCurrentEvent(*ep);
	assert(ev != 0);
      }
      assert(ev == ep.get());
      result.push_back(ep);
      --eventsRemainingInFile_;
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readManySequential(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    result.reserve(number);
    skipBadFiles_ = false;
    if (fileIter_ == fileIterEnd_ || !rootFile_) {
      fileIter_ = fileIterBegin_;
      initFile(false);
      rootFile_->setAtEventEntry(0);
    }
    fileSeqNumber = fileIter_ - fileIterBegin_;
    unsigned int numberRead = 0;
    for(int i = 0; i < number; ++i) {
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(rootFile_->productRegistry(), processConfiguration()));
      EventPrincipal* ev = rootFile_->readCurrentEvent(*ep);
      if(ev == 0) {
	if (numberRead == 0) {
	  ++fileIter_;
          fileSeqNumber = fileIter_ - fileIterBegin_;
	  if (fileIter_ == fileIterEnd_) {
	    return;
	  }
	  initFile(false);
	  rootFile_->setAtEventEntry(0);
	  return readManySequential(number, result, fileSeqNumber);
	}
	return;
      }
      assert(ev == ep.get());
      result.push_back(ep);
      ++numberRead;
      rootFile_->nextEventEntry();
    }
  }

  void
  RootInputFileSequence::readManySpecified(std::vector<EventID> const& events, EventPrincipalVector& result) {
    skipBadFiles_ = false;
    result.reserve(events.size());
    for (std::vector<EventID>::const_iterator it = events.begin(), itEnd = events.end(); it != itEnd; ++it) {
      bool found = skipToItem(it->run(), it->luminosityBlock(), it->event(), true, false);
      if (!found) {
	throw edm::Exception(edm::errors::NotFound) <<
	   "RootInputFileSequence::readManySpecified_(): Secondary Input file " <<
	   fileIter_->fileName() <<
           " does not contain specified event:\n" << *it << "\n";
      }
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(rootFile_->productRegistry(), processConfiguration()));
      EventPrincipal* ev = rootFile_->readCurrentEvent(*ep);
      if (ev == 0) {
	throw edm::Exception(edm::errors::EventCorruption) <<
	   "RootInputFileSequence::readManySpecified_(): Secondary Input file " <<
	   fileIter_->fileName() <<
           " contains specified event " << *it << " that cannot be read.\n";
      }
      assert(ev == ep.get());
      result.push_back(ep);
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

  void
  RootInputFileSequence::fillDescription(ParameterSetDescription & desc) {
    desc.addUntracked<unsigned int>("skipEvents", 0U);
    desc.addUntracked<bool>("noEventSort", false);
    desc.addUntracked<bool>("skipBadFiles", false);
    desc.addUntracked<unsigned int>("cacheSize", input::defaultCacheSize);
    desc.addUntracked<int>("treeMaxVirtualSize", -1);
    desc.addUntracked<unsigned int>("setRunNumber", 0U);
    desc.addUntracked<bool>("dropDescendantsOfDroppedBranches", true);

    std::string defaultString("permissive");
    desc.addUntracked<std::string>("parametersMustMatch", defaultString);
    desc.addUntracked<std::string>("branchesMustMatch", defaultString);

    std::vector<std::string> defaultStrings(1U, std::string("keep *"));
    desc.addUntracked<std::vector<std::string> >("inputCommands", defaultStrings);

    EventSkipperByID::fillDescription(desc);
    DuplicateChecker::fillDescription(desc);
  }
}


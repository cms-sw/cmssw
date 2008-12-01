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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

#include "CLHEP/Random/RandFlat.h"
#include "TFile.h"

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
    matchMode_(BranchDescription::Permissive),
    flatDistribution_(0),
    fileIndexes_(fileCatalogItems().size()),
    eventsRemainingInFile_(0),
    startAtRun_(pset.getUntrackedParameter<unsigned int>("firstRun", 1U)),
    startAtLumi_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1U)),
    startAtEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1U)),
    eventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0U)),
    whichLumisToSkip_(pset.getUntrackedParameter<std::vector<LuminosityBlockID> >("lumisToSkip", std::vector<LuminosityBlockID>())),
    eventsToProcess_(pset.getUntrackedParameter<std::vector<EventID> >("eventsToProcess",std::vector<EventID>())),
    noEventSort_(pset.getUntrackedParameter<bool>("noEventSort", false)),
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
    treeCacheSize_(pset.getUntrackedParameter<unsigned int>("cacheSize", 0U)),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    forcedRunOffset_(0),
    setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber", 0U)),
    groupSelectorRules_(pset, "inputCommands", "InputSource"),
    dropMetaData_(pset.getUntrackedParameter<bool>("dropMetaData", false)),
    primarySequence_(primarySequence),
    randomAccess_(false),
    duplicateChecker_(),
    dropDescendents_(pset.getUntrackedParameter<bool>("dropDescendentsOfDroppedBranches", true)) {

    if (!primarySequence_) noEventSort_ = false;
    if (noEventSort_ && ((startAtEvent_ > 1) || !eventsToProcess_.empty())) {
      throw edm::Exception(errors::Configuration)
        << "Illegal configuration options passed to PoolSource\n"
        << "You cannot request \"noEventSort\" and also set \"firstEvent\"\n"
        << "or \"eventsToProcess\".\n";
    }

    if (primarySequence_ && primary()) duplicateChecker_.reset(new DuplicateChecker(pset));

    StorageFactory *factory = StorageFactory::get();
    for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_)
      factory->stagein(fileIter_->fileName());

    sort_all(eventsToProcess_);
    std::string matchMode = pset.getUntrackedParameter<std::string>("fileMatchMode", std::string("permissive"));
    if (matchMode == std::string("strict")) matchMode_ = BranchDescription::Strict;
    if (primary()) {
      for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_) {
        initFile(skipBadFiles_);
        if (rootFile_) break;
      }
      if (rootFile_) {
        forcedRunOffset_ = rootFile_->setForcedRunOffset(setRun_);
        if (forcedRunOffset_ < 0) {
          throw edm::Exception(errors::Configuration)
            << "The value of the 'setRunNumber' parameter must not be\n"
            << "less than the first run number in the first input file.\n"
            << "'setRunNumber' was " << setRun_ <<", while the first run was "
            << setRun_ - forcedRunOffset_ << ".\n";
        }
        updateProductRegistry();
      }
    } else {
      Service<RandomNumberGenerator> rng;
      if (!rng.isAvailable()) {
        throw edm::Exception(errors::Configuration)
          << "A secondary input source requires the RandomNumberGeneratorService\n"
          << "which is not present in the configuration file.  You must add the service\n"
          << "in the configuration file or remove the modules that require it.";
      }
      CLHEP::HepRandomEngine& engine = rng->getEngine();
      flatDistribution_ = new CLHEP::RandFlat(engine);
    }
  }

  std::vector<FileCatalogItem> const&
  RootInputFileSequence::fileCatalogItems() const {
    return catalog_.fileCatalogItems();
  }

  void
  RootInputFileSequence::endJob() {
    closeFile_();
  }

  boost::shared_ptr<FileBlock>
  RootInputFileSequence::readFile_() {
    if (firstFile_) {
      // The first input file has already been opened, or a rewind has occurred.
      firstFile_ = false;
      if (!rootFile_) {
	initFile(skipBadFiles_);
      }
    } else {
      if (!nextFile()) {
        assert(0);
      }
    }
    if (!rootFile_) {
      return boost::shared_ptr<FileBlock>(new FileBlock);
    }
    if (primary()) {
      productRegistryUpdate().setProductIDs(rootFile_->productRegistry()->nextID());
      if (rootFile_->productRegistry()->nextID() > productRegistry()->nextID()) {
        productRegistryUpdate().setNextID(rootFile_->productRegistry()->nextID());
      }
    }
    return rootFile_->createFileBlock();
  }

  void RootInputFileSequence::closeFile_() {
    if (rootFile_) {
    // Account for events skipped in the file.
      eventsToSkip_ = rootFile_->eventsToSkip();
      {
        std::auto_ptr<InputSource::FileCloseSentry> 
	  sentry((primarySequence_ && primary()) ? new InputSource::FileCloseSentry(input_) : 0);
        rootFile_->close(primary());
      }
      logFileAction("  Closed file ", rootFile_->file());
      rootFile_.reset();
      if (duplicateChecker_.get() != 0) duplicateChecker_->inputFileClosed();
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
      filePtr.reset(TFile::Open(fileIter_->fileName().c_str()));
    }
    catch (cms::Exception e) {
      if (!skipBadFiles) {
	throw edm::Exception(edm::errors::FileOpenError) << e.explainSelf() << "\n" <<
	   "RootInputFileSequence::initFile(): Input file " << fileIter_->fileName() << " was not found or could not be opened.\n";
      }
    }
    if (filePtr && !filePtr->IsZombie()) {
      logFileAction("  Successfully opened file ", fileIter_->fileName());
      rootFile_ = RootFileSharedPtr(new RootFile(fileIter_->fileName(), catalog_.url(),
	  processConfiguration(), fileIter_->logicalFileName(), filePtr,
	  startAtRun_, startAtLumi_, startAtEvent_, eventsToSkip_, whichLumisToSkip_,
	  remainingEvents(), remainingLuminosityBlocks(), treeCacheSize_, treeMaxVirtualSize_,
	  input_.processingMode(),
	  forcedRunOffset_, eventsToProcess_, noEventSort_,
	  dropMetaData_, groupSelectorRules_, !primarySequence_, duplicateChecker_, dropDescendents_));
      fileIndexes_[fileIter_ - fileIterBegin_] = rootFile_->fileIndexSharedPtr();
    } else {
      if (!skipBadFiles) {
	throw edm::Exception(edm::errors::FileOpenError) <<
	   "RootInputFileSequence::initFile(): Input file " << fileIter_->fileName() << " was not found or could not be opened.\n";
      }
      LogWarning("") << "Input file: " << fileIter_->fileName() << " was not found or could not be opened, and will be skipped.\n";
    }
  }

  void RootInputFileSequence::updateProductRegistry() const {
    ProductRegistry::ProductList const& prodList = rootFile_->productRegistry()->productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
	it != itEnd; ++it) {
      productRegistryUpdate().copyProduct(it->second);
    }
  }

  ProductRegistry const&
  RootInputFileSequence::fileProductRegistry() const {
    return *rootFile_->productRegistry();
  }

  bool RootInputFileSequence::nextFile() {
    if(fileIter_ != fileIterEnd_) ++fileIter_;
    if(fileIter_ == fileIterEnd_) {
      if (primarySequence_) {
	return false;
      } else {
	fileIter_ = fileIterBegin_;
      }
    }

    initFile(skipBadFiles_);

    if (primarySequence_ && rootFile_) {
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    matchMode_);
      if (!mergeInfo.empty()) {
        throw edm::Exception(errors::MismatchedInputFiles,"RootInputFileSequence::nextFile()") << mergeInfo;
      }
    }
    return true;
  }

  bool RootInputFileSequence::previousFile() {
    if(fileIter_ == fileIterBegin_) {
      if (primarySequence_) {
	return false;
      } else {
	fileIter_ = fileIterEnd_;
      }
    }
    --fileIter_;

    initFile(false);

    if (primarySequence_ && rootFile_) {
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    matchMode_);
      if (!mergeInfo.empty()) {
        throw edm::Exception(errors::MismatchedInputFiles,"RootInputFileSequence::previousEvent()") << mergeInfo;
      }
    }
    if (rootFile_) rootFile_->setToLastEntry();
    return true;
  }

  RootInputFileSequence::~RootInputFileSequence() {
  }

  boost::shared_ptr<RunPrincipal>
  RootInputFileSequence::readRun_() {
    return rootFile_->readRun(primarySequence_ ? productRegistry() : rootFile_->productRegistry()); 
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootInputFileSequence::readLuminosityBlock_() {
    return rootFile_->readLumi(primarySequence_ ? productRegistry() : rootFile_->productRegistry(), runPrincipal()); 
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

  std::auto_ptr<EventPrincipal>
  RootInputFileSequence::readIt(EventID const& id, LuminosityBlockNumber_t lumi, bool exact) {
    randomAccess_ = true;
    // Attempt to find event in currently open input file.
    bool found = rootFile_->setEntryAtEvent(id.run(), lumi, id.event(), exact);
    if (!found) {
      // If only one input file, give up now, to save time.
      if (fileIndexes_.size() == 1) {
	return std::auto_ptr<EventPrincipal>(0);
      }
      // Look for event in files previously opened without reopening unnecessary files.
      typedef std::vector<boost::shared_ptr<FileIndex> >::const_iterator Iter;
      for (Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if (*it && (*it)->containsEvent(id.run(), lumi, id.event(), exact)) {
          // We found it. Close the currently open file, and open the correct one.
	  fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
	  // Now get the event from the correct file.
          found = rootFile_->setEntryAtEvent(id.run(), lumi, id.event(), exact);
	  assert (found);
	  std::auto_ptr<EventPrincipal> ep = readCurrentEvent();
          skip(1);
	  return ep;
	}
      }
      // Look for event in files not yet opened.
      for (Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if (!*it) {
	  fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
          found = rootFile_->setEntryAtEvent(id.run(), lumi, id.event(), exact);
	  if (found) {
	    std::auto_ptr<EventPrincipal> ep = readCurrentEvent();
            skip(1);
	    return ep;
	  }
	}
      }
      // Not found
      return std::auto_ptr<EventPrincipal>(0);
    }
    std::auto_ptr<EventPrincipal> eptr = readCurrentEvent();
    skip(1);
    return eptr;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootInputFileSequence::readIt(LuminosityBlockID const& id) {

    // Attempt to find lumi in currently open input file.
    bool found = rootFile_->setEntryAtLumi(id);
    if (found) {
      return readLuminosityBlock_();
    }

    if (fileIndexes_.size() > 1) {
      // Look for lumi in files previously opened without reopening unnecessary files.
      typedef std::vector<boost::shared_ptr<FileIndex> >::const_iterator Iter;
      for (Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if (*it && (*it)->containsLumi(id.run(), id.luminosityBlock(), true)) {
          // We found it. Close the currently open file, and open the correct one.
          fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
	  // Now get the lumi from the correct file.
          found = rootFile_->setEntryAtLumi(id);
	  assert (found);
          return readLuminosityBlock_();
	}
      }
      // Look for lumi in files not yet opened.
      for (Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if (!*it) {
          fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
          found = rootFile_->setEntryAtLumi(id);
	  if (found) {
            return readLuminosityBlock_();
	  }
	}
      }
    }
    return boost::shared_ptr<LuminosityBlockPrincipal>();
  }

  boost::shared_ptr<RunPrincipal>
  RootInputFileSequence::readIt(RunID const& id) {

    // Attempt to find run in currently open input file.
    bool found = rootFile_->setEntryAtRun(id);
    if (found) {
      return readRun_();
    }
    if (fileIndexes_.size() > 1) {
      // Look for run in files previously opened without reopening unnecessary files.
      typedef std::vector<boost::shared_ptr<FileIndex> >::const_iterator Iter;
      for (Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if (*it && (*it)->containsRun(id.run(), true)) {
          // We found it. Close the currently open file, and open the correct one.
          fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
	  // Now get the event from the correct file.
          found = rootFile_->setEntryAtRun(id);
	  assert (found);
          return readRun_();
	}
      }
      // Look for run in files not yet opened.
      for (Iter it = fileIndexes_.begin(), itEnd = fileIndexes_.end(); it != itEnd; ++it) {
	if (!*it) {
          fileIter_ = fileIterBegin_ + (it - fileIndexes_.begin());
	  initFile(false);
          found = rootFile_->setEntryAtRun(id);
	  if (found) {
            return readRun_();
	  }
	}
      }
    }
    return boost::shared_ptr<RunPrincipal>();
  }

  InputSource::ItemType
  RootInputFileSequence::getNextItemType() {
    if (fileIter_ == fileIterEnd_) {
      return InputSource::IsStop;
    }
    if (firstFile_) {
      return InputSource::IsFile;
    }
    if (rootFile_) {
      if (randomAccess_) {
        skip(0);
        if (fileIter_== fileIterEnd_) {
          return InputSource::IsStop;
        }
      }
      FileIndex::EntryType entryType = rootFile_->getNextEntryTypeWanted();
      if (entryType == FileIndex::kEvent) {
        return InputSource::IsEvent;
      } else if (entryType == FileIndex::kLumi) {
        return InputSource::IsLumi;
      } else if (entryType == FileIndex::kRun) {
        return InputSource::IsRun;
      }
      assert(entryType == FileIndex::kEnd);
    }
    if (fileIter_ + 1 == fileIterEnd_) {
      return InputSource::IsStop;
    }
    return InputSource::IsFile;
  }

  // Rewind to before the first event that was read.
  void
  RootInputFileSequence::rewind_() {
    randomAccess_ = false;
    firstFile_ = true;
    fileIter_ = fileIterBegin_;
    if (duplicateChecker_.get() != 0) duplicateChecker_->rewind();
  }

  // Rewind to the beginning of the current file
  void
  RootInputFileSequence::rewindFile() {
    rootFile_->rewind();
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void
  RootInputFileSequence::skip(int offset) {
    randomAccess_ = true;
    while (offset != 0) {
      offset = rootFile_->skipEvents(offset);
      if (offset > 0 && !nextFile()) return;
      if (offset < 0 && !previousFile()) return;
    }
    rootFile_->skipEvents(0);
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
    for (std::vector<std::string>::const_iterator it = wantedBranches.begin(), itEnd = wantedBranches.end();
	it != itEnd; ++it) {
      rules.push_back("keep " + *it + "_*");
    }
    ParameterSet pset;
    pset.addUntrackedParameter("inputCommands", rules);
    groupSelectorRules_ = GroupSelectorRules(pset, "inputCommands", "InputSource");
  }

  void
  RootInputFileSequence::readMany_(int number, EventPrincipalVector& result) {
    for (int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if (ev.get() == 0) {
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
    if (currentSeqNumber != fileSeqNumber) {
      fileIter_ = fileIterBegin_ + fileSeqNumber;
      initFile(false);
    }
    rootFile_->setEntryAtEvent(id.run(), 0U, id.event(), false);
    for (int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if (ev.get() == 0) {
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
    skipBadFiles_ = false;
    unsigned int currentSeqNumber = fileIter_ - fileIterBegin_;
    while (eventsRemainingInFile_ < number) {
      fileIter_ = fileIterBegin_ + flatDistribution_->fireInt(fileCatalogItems().size());
      unsigned int newSeqNumber = fileIter_ - fileIterBegin_;
      if (newSeqNumber != currentSeqNumber) {
        initFile(false);
      }
      eventsRemainingInFile_ = rootFile_->eventTree().entries();
      if (eventsRemainingInFile_ == 0) {
	throw edm::Exception(edm::errors::NotFound) <<
	   "RootInputFileSequence::readManyRandom_(): Secondary Input file " << fileIter_->fileName() << " contains no events.\n";
      }
      rootFile_->setAtEventEntry(flatDistribution_->fireInt(eventsRemainingInFile_));
    }
    fileSeqNumber = fileIter_ - fileIterBegin_;
    for (int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readCurrentEvent();
      if (ev.get() == 0) {
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

  void RootInputFileSequence::logFileAction(const char* msg, std::string const& file) {
    if (primarySequence_) {
      time_t t = time(0);
      char ts[] = "dd-Mon-yyyy hh:mm:ss TZN     ";
      strftime( ts, strlen(ts)+1, "%d-%b-%Y %H:%M:%S %Z", localtime(&t) );
      edm::LogAbsolute("fileAction") << ts << msg << file;
      edm::FlushMessageLog();
    }
  }
}


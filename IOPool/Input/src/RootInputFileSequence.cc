/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DuplicateChecker.h"
#include "PoolSource.h"
#include "RootFile.h"
#include "RootInputFileSequence.h"
#include "RootTree.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

#include "CLHEP/Random/RandFlat.h"
#include "InputFile.h"
#include "TSystem.h"

namespace edm {
  RootInputFileSequence::RootInputFileSequence(
                ParameterSet const& pset,
                PoolSource const& input,
                InputFileCatalog const& catalog,
                PrincipalCache& cache,
                InputType::InputType inputType) :
    input_(input),
    inputType_(inputType),
    catalog_(catalog),
    firstFile_(true),
    fileIterBegin_(fileCatalogItems().begin()),
    fileIterEnd_(fileCatalogItems().end()),
    fileIter_(fileIterEnd_),
    fileIterLastOpened_(fileIterEnd_),
    rootFile_(),
    parametersMustMatch_(BranchDescription::Permissive),
    branchesMustMatch_(BranchDescription::Permissive),
    flatDistribution_(),
    indexesIntoFiles_(fileCatalogItems().size()),
    orderedProcessHistoryIDs_(),
    eventSkipperByID_(inputType == InputType::Primary ? EventSkipperByID::create(pset).release() : 0),
    eventsRemainingInFile_(0),
    // The default value provided as the second argument to the getUntrackedParameter function call
    // is not used when the ParameterSet has been validated and the parameters are not optional
    // in the description.  This is currently true when PoolSource is the primary input source.
    // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
    // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
    // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
    // and should be deleted from the code.
    numberOfEventsToSkip_(inputType == InputType::Primary ? pset.getUntrackedParameter<unsigned int>("skipEvents", 0U) : 0U),
    noEventSort_(inputType == InputType::Primary ? pset.getUntrackedParameter<bool>("noEventSort", true) : false),
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
    treeCacheSize_(noEventSort_ ? pset.getUntrackedParameter<unsigned int>("cacheSize", roottree::defaultCacheSize) : 0U),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber", 0U)),
    groupSelectorRules_(pset, "inputCommands", "InputSource"),
    duplicateChecker_(inputType == InputType::Primary ? new DuplicateChecker(pset) : 0),
    dropDescendants_(pset.getUntrackedParameter<bool>("dropDescendantsOfDroppedBranches", inputType != InputType::SecondarySource)),
    usingGoToEvent_(false) {

    //we now allow the site local config to specify what the TTree cache size should be
    Service<SiteLocalConfig> pSLC;
    if(treeCacheSize_ != 0U && pSLC.isAvailable() && pSLC->sourceTTreeCacheSize()) {
      treeCacheSize_ = *(pSLC->sourceTTreeCacheSize());
    }

    if(inputType_ == InputType::Primary) {
      //NOTE: we do not want to stage in secondary files since we can be given a list of
      // thousands of files and prestaging all those files can cause a site to fail
      StorageFactory *factory = StorageFactory::get();
      for(fileIter_ = fileIterBegin_; fileIter_ != fileIterEnd_; ++fileIter_) {
        factory->activateTimeout(fileIter_->fileName());
        factory->stagein(fileIter_->fileName());
      }
    }

    std::string parametersMustMatch = pset.getUntrackedParameter<std::string>("parametersMustMatch", std::string("permissive"));
    if(parametersMustMatch == std::string("strict")) parametersMustMatch_ = BranchDescription::Strict;

    std::string branchesMustMatch = pset.getUntrackedParameter<std::string>("branchesMustMatch", std::string("permissive"));
    if(branchesMustMatch == std::string("strict")) branchesMustMatch_ = BranchDescription::Strict;

    if(inputType != InputType::SecondarySource) {
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
    closeFile_();
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
    // close the currently open file, if any, and delete the RootFile object.
    if(rootFile_) {
      if(inputType_ != InputType::SecondarySource) {
        std::auto_ptr<InputSource::FileCloseSentry>
        sentry((inputType_ == InputType::Primary) ? new InputSource::FileCloseSentry(input_) : 0);
        rootFile_->close();
        if(duplicateChecker_) duplicateChecker_->inputFileClosed();
      }
      rootFile_.reset();
    }
  }

  void RootInputFileSequence::initFile(bool skipBadFiles) {
    // We are really going to close the open file.

    // If this is the primary sequence, we are not duplicate checking across files
    // and we are not using random access to find events, then we can delete the
    // IndexIntoFile for the file we are closing. If we can't delete all of it,
    // then we can delete the parts we do not need.
    if(fileIterLastOpened_ != fileIterEnd_) {
      size_t currentIndexIntoFile = fileIterLastOpened_ - fileIterBegin_;
      bool needIndexesForDuplicateChecker = duplicateChecker_ && duplicateChecker_->checkingAllFiles() && !duplicateChecker_->checkDisabled();
      bool deleteIndexIntoFile = inputType_ == InputType::Primary &&
                                 !needIndexesForDuplicateChecker &&
                                 !usingGoToEvent_;
      if(deleteIndexIntoFile) {
        indexesIntoFiles_[currentIndexIntoFile].reset();
      } else {
        if(indexesIntoFiles_[currentIndexIntoFile]) indexesIntoFiles_[currentIndexIntoFile]->inputFileClosed();
      }
      fileIterLastOpened_ = fileIterEnd_;
    }
    closeFile_();

    // Check if the logical file name was found.
    if(fileIter_->fileName().empty()) {
      // LFN not found in catalog.
      InputFile::reportSkippedFile(fileIter_->fileName(), fileIter_->logicalFileName());
      if(!skipBadFiles) {
        throw cms::Exception("LogicalFileNameNotFound", "RootInputFileSequence::initFile()\n")
          << "Logical file name '" << fileIter_->logicalFileName() << "' was not found in the file catalog.\n"
          << "If you wanted a local file, you forgot the 'file:' prefix\n"
          << "before the file name in your configuration file.\n";
      }
      LogWarning("") << "Input logical file: " << fileIter_->logicalFileName() << " was not found in the catalog, and will be skipped.\n";
      return;
    }

    // Determine whether we have a fallback URL specified; if so, prepare it;
    // Only valid if it is non-empty and differs from the original filename.
    std::string fallbackName = fileIter_->fallbackFileName();
    bool hasFallbackUrl = !fallbackName.empty() && fallbackName != fileIter_->fileName();

    boost::shared_ptr<InputFile> filePtr;
    try {
      std::auto_ptr<InputSource::FileOpenSentry>
        sentry(inputType_ == InputType::Primary ? new InputSource::FileOpenSentry(input_) : 0);
      filePtr.reset(new InputFile(gSystem->ExpandPathName(fileIter_->fileName().c_str()), "  Initiating request to open file "));
    }
    catch (cms::Exception const& e) {
      if(!skipBadFiles && !hasFallbackUrl) {
        InputFile::reportSkippedFile(fileIter_->fileName(), fileIter_->logicalFileName());
        Exception ex(errors::FileOpenError, "", e);
        ex.addContext("Calling RootInputFileSequence::initFile()");
        //ex.clearMessage();
        ex << "Input file " << fileIter_->fileName() << " was not found, could not be opened, or is corrupted.\n";
        throw ex;
      }
    }
    if(!filePtr && (hasFallbackUrl)) {
      try {
        std::auto_ptr<InputSource::FileOpenSentry>
          sentry(inputType_ == InputType::Primary ? new InputSource::FileOpenSentry(input_) : 0);
        filePtr.reset(new InputFile(gSystem->ExpandPathName(fallbackName.c_str()), "  Fallback request to file "));
      }
      catch (cms::Exception const& e) {
        if(!skipBadFiles) {
          InputFile::reportSkippedFile(fileIter_->fileName(), fileIter_->logicalFileName());
          Exception ex(errors::FileOpenError, "", e);
          ex.addContext("Calling RootInputFileSequence::initFile()");
          //ex.clearMessage();
          ex << "Input file " << fileIter_->fileName() << " was not found, could not be opened, or is corrupted.\n";
          ex << "Fallback Input file " << fallbackName << " also was not found, could not be opened, or is corrupted.\n";
          throw ex;
        }
      }
    }
    if(filePtr) {
      std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile = fileIter_ - fileIterBegin_;
      rootFile_ = RootFileSharedPtr(new RootFile(fileIter_->fileName(),
          processConfiguration(), fileIter_->logicalFileName(), filePtr,
          eventSkipperByID_, numberOfEventsToSkip_ != 0,
          remainingEvents(), remainingLuminosityBlocks(), treeCacheSize_, treeMaxVirtualSize_,
          input_.processingMode(),
          setRun_,
          noEventSort_,
          groupSelectorRules_,
          inputType_,
          duplicateChecker_,
          dropDescendants_,
          indexesIntoFiles_, currentIndexIntoFile, orderedProcessHistoryIDs_, usingGoToEvent_));

      fileIterLastOpened_ = fileIter_;
      indexesIntoFiles_[currentIndexIntoFile] = rootFile_->indexIntoFileSharedPtr();
      char const* inputType = 0;
      switch(inputType_) {
      case InputType::Primary: inputType = "primaryFiles"; break;
      case InputType::SecondaryFile: inputType = "secondaryFiles"; break;
      case InputType::SecondarySource: inputType = "mixingFiles"; break;
      }
      rootFile_->reportOpened(inputType);
    } else {
      InputFile::reportSkippedFile(fileIter_->fileName(), fileIter_->logicalFileName());
      if(!skipBadFiles) {
        throw Exception(errors::FileOpenError) <<
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
      if(inputType_ == InputType::Primary) {
        return false;
      } else {
        fileIter_ = fileIterBegin_;
      }
    }

    initFile(skipBadFiles_);

    if(inputType_ == InputType::Primary && rootFile_) {
      size_t size = productRegistry()->size();
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
                                                            fileIter_->fileName(),
                                                            parametersMustMatch_,
                                                            branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw Exception(errors::MismatchedInputFiles,"RootInputFileSequence::nextFile()") << mergeInfo;
      }
      if(productRegistry()->size() > size) {
        cache.adjustIndexesAfterProductRegistryAddition();
      }
      cache.adjustEventToNewProductRegistry(productRegistry());
    }
    return true;
  }

  bool RootInputFileSequence::previousFile(PrincipalCache& cache) {
    if(fileIter_ == fileIterBegin_) {
      if(inputType_ == InputType::Primary) {
        return false;
      } else {
        fileIter_ = fileIterEnd_;
      }
    }
    --fileIter_;

    initFile(false);

    if(inputType_ == InputType::Primary && rootFile_) {
      size_t size = productRegistry()->size();
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
                                                            fileIter_->fileName(),
                                                            parametersMustMatch_,
                                                            branchesMustMatch_);
      if(!mergeInfo.empty()) {
        throw Exception(errors::MismatchedInputFiles,"RootInputFileSequence::previousEvent()") << mergeInfo;
      }
      if(productRegistry()->size() > size) {
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
    return rootFile_->readRunAuxiliary_();
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  RootInputFileSequence::readLuminosityBlockAuxiliary_() {
    return rootFile_->readLuminosityBlockAuxiliary_();
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
      IndexIntoFile::EntryType entryType = rootFile_->getNextEntryTypeWanted();
      if(entryType == IndexIntoFile::kEvent) {
        return InputSource::IsEvent;
      } else if(entryType == IndexIntoFile::kLumi) {
        return InputSource::IsLumi;
      } else if(entryType == IndexIntoFile::kRun) {
        return InputSource::IsRun;
      }
      assert(entryType == IndexIntoFile::kEnd);
    }
    if(fileIter_ + 1 == fileIterEnd_) {
      return InputSource::IsStop;
    }
    return InputSource::IsFile;
  }

  // Rewind to before the first event that was read.
  void
  RootInputFileSequence::rewind_() {
    if(fileIter_ != fileIterBegin_) {
      closeFile_();
      fileIter_ = fileIterBegin_;
    }
    if(!rootFile_) {
      initFile(false);
    }
    rewindFile();
    firstFile_ = true;
  }

  // Rewind to the beginning of the current file
  void
  RootInputFileSequence::rewindFile() {
    rootFile_->rewind();
  }

  void
  RootInputFileSequence::reset(PrincipalCache& cache) {
    //NOTE: Need to handle duplicate checker
    // Also what if skipBadFiles_==true and the first time we succeeded but after a reset we fail?
    if(inputType_ != InputType::SecondarySource) {
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
        fileIter_ = fileIterEnd_;
        return false;
      }
    }
    return true;
  }

  bool
  RootInputFileSequence::goToEvent(EventID const& eventID) {
    usingGoToEvent_ = true;
    if(rootFile_) {
      if(rootFile_->goToEvent(eventID)) {
        return true;
      }
      // If only one input file, give up now, to save time.
      if(rootFile_ && indexesIntoFiles_.size() == 1) {
        return false;
      }
      // Save the current file and position so that we can restore them
      // if we fail to restore the desired event
      bool closedOriginalFile = false;
      std::vector<FileCatalogItem>::const_iterator originalFile = fileIter_;
      IndexIntoFile::IndexIntoFileItr originalPosition = rootFile_->indexIntoFileIter();

      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      typedef std::vector<boost::shared_ptr<IndexIntoFile> >::const_iterator Iter;
      for(Iter it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
        if(*it && (*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
          // We found it. Close the currently open file, and open the correct one.
          fileIter_ = fileIterBegin_ + (it - indexesIntoFiles_.begin());
          initFile(false);
          // Now get the item from the correct file.
          bool found = rootFile_->goToEvent(eventID);
          assert (found);
          return true;
        }
      }
      // Look for item in files not yet opened.
      for(Iter it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
        if(!*it) {
          fileIter_ = fileIterBegin_ + (it - indexesIntoFiles_.begin());
          initFile(false);
          closedOriginalFile = true;
          if((*it)->containsItem(eventID.run(), eventID.luminosityBlock(), eventID.event())) {
            if  (rootFile_->goToEvent(eventID)) {
              return true;
            }
          }
        }
      }
      if(closedOriginalFile) {
        fileIter_ = originalFile;
        initFile(false);
        rootFile_->setPosition(originalPosition);
      }
    }
    return false;
  }

  bool
  RootInputFileSequence::skipToItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
    // Attempt to find item in currently open input file.
    bool found = rootFile_ && rootFile_->setEntryAtItem(run, lumi, event);
    if(!found) {
      // If only one input file, give up now, to save time.
      if(rootFile_ && indexesIntoFiles_.size() == 1) {
        return false;
      }
      // Look for item (run/lumi/event) in files previously opened without reopening unnecessary files.
      typedef std::vector<boost::shared_ptr<IndexIntoFile> >::const_iterator Iter;
      for(Iter it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
        if(*it && (*it)->containsItem(run, lumi, event)) {
          // We found it. Close the currently open file, and open the correct one.
          fileIter_ = fileIterBegin_ + (it - indexesIntoFiles_.begin());
          initFile(false);
          // Now get the item from the correct file.
          found = rootFile_->setEntryAtItem(run, lumi, event);
          assert (found);
          return true;
        }
      }
      // Look for item in files not yet opened.
      for(Iter it = indexesIntoFiles_.begin(), itEnd = indexesIntoFiles_.end(); it != itEnd; ++it) {
        if(!*it) {
          fileIter_ = fileIterBegin_ + (it - indexesIntoFiles_.begin());
          initFile(false);
          found = rootFile_->setEntryAtItem(run, lumi, event);
          if(found) {
            return true;
          }
        }
      }
      // Not found
      return false;
    }
    return true;
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

  EventPrincipal*
  RootInputFileSequence::readOneSequential() {
    skipBadFiles_ = false;
    if(fileIter_ == fileIterEnd_ || !rootFile_) {
      if(fileIterEnd_ == fileIterBegin_) {
        throw Exception(errors::Configuration) << "RootInputFileSequence::readOneSequential(): no input files specified.\n";
      }
      fileIter_ = fileIterBegin_;
      initFile(false);
      rootFile_->setAtEventEntry(-1);
    }
    rootFile_->nextEventEntry();
    EventPrincipal* ep = rootFile_->clearAndReadCurrentEvent(rootFile_->secondaryEventPrincipal());
    if(ep == 0) {
      ++fileIter_;
      if(fileIter_ == fileIterEnd_) {
        return 0;
      }
      initFile(false);
      rootFile_->setAtEventEntry(-1);
      return readOneSequential();
    }
    return ep;
  }

  EventPrincipal*
  RootInputFileSequence::readOneSpecified(EventID const& id) {
    skipBadFiles_ = false;
    bool found = skipToItem(id.run(), id.luminosityBlock(), id.event());
    if(!found) {
      throw Exception(errors::NotFound) <<
         "RootInputFileSequence::readOneSpecified(): Secondary Input file " <<
         fileIter_->fileName() <<
         " does not contain specified event:\n" << id << "\n";
    }
    EventPrincipal* ep = rootFile_->clearAndReadCurrentEvent(rootFile_->secondaryEventPrincipal());
    assert(ep != 0);
    return ep;
  }

  EventPrincipal*
  RootInputFileSequence::readOneRandom() {
    if(fileIterEnd_ == fileIterBegin_) {
      throw Exception(errors::Configuration) << "RootInputFileSequence::readOneRandom(): no input files specified.\n";
    }
    if(!flatDistribution_) {
      Service<RandomNumberGenerator> rng;
      CLHEP::HepRandomEngine& engine = rng->getEngine();
      flatDistribution_.reset(new CLHEP::RandFlat(engine));
    }
    skipBadFiles_ = false;
    unsigned int currentSeqNumber = fileIter_ - fileIterBegin_;
    while(eventsRemainingInFile_ == 0) {
      fileIter_ = fileIterBegin_ + flatDistribution_->fireInt(fileCatalogItems().size());
      unsigned int newSeqNumber = fileIter_ - fileIterBegin_;
      if(newSeqNumber != currentSeqNumber) {
        initFile(false);
        currentSeqNumber = newSeqNumber;
      }
      eventsRemainingInFile_ = rootFile_->eventTree().entries();
      if(eventsRemainingInFile_ == 0) {
        throw Exception(errors::NotFound) <<
           "RootInputFileSequence::readOneRandom(): Secondary Input file " << fileIter_->fileName() << " contains no events.\n";
      }
      rootFile_->setAtEventEntry(flatDistribution_->fireInt(eventsRemainingInFile_) - 1);
    }
    rootFile_->nextEventEntry();

    EventPrincipal* ep = rootFile_->clearAndReadCurrentEvent(rootFile_->secondaryEventPrincipal());
    if(ep == 0) {
      rootFile_->setAtEventEntry(0);
      ep = rootFile_->clearAndReadCurrentEvent(rootFile_->secondaryEventPrincipal());
      assert(ep != 0);
    }
    --eventsRemainingInFile_;
    return ep;
  }

  void
  RootInputFileSequence::fillDescription(ParameterSetDescription & desc) {
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
    desc.addUntracked<unsigned int>("cacheSize", roottree::defaultCacheSize)
        ->setComment("Size of ROOT TTree prefetch cache.  Affects performance.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache.  Affects performance.");
    desc.addUntracked<unsigned int>("setRunNumber", 0U)
        ->setComment("If non-zero, change number of first run to this number. Apply same offset to all runs.  Allowed only for simulation.");
    desc.addUntracked<bool>("dropDescendantsOfDroppedBranches", true)
        ->setComment("If True, also drop on input any descendent of any branch dropped on input.");
    std::string defaultString("permissive");
    desc.addUntracked<std::string>("parametersMustMatch", defaultString)
        ->setComment("'strict':     Values of tracked parameters must be unique across all input files.\n"
                     "'permissive': Values of tracked parameters may differ across or within files.");
    desc.addUntracked<std::string>("branchesMustMatch", defaultString)
        ->setComment("'strict':     Branches in each input file must match those in the first file.\n"
                     "'permissive': Branches in each input file may be any subset of those in the first file.");

    GroupSelectorRules::fillDescription(desc, "inputCommands");
    EventSkipperByID::fillDescription(desc);
    DuplicateChecker::fillDescription(desc);
  }

  ProcessingController::ForwardState
  RootInputFileSequence::forwardState() const {
    if(rootFile_) {
      if(!rootFile_->wasLastEventJustRead()) {
        return ProcessingController::kEventsAheadInFile;
      }
      std::vector<FileCatalogItem>::const_iterator itr(fileIter_);
      if(itr != fileIterEnd_) ++itr;
      if(itr != fileIterEnd_) {
        return ProcessingController::kNextFileExists;
      }
      return ProcessingController::kAtLastEvent;
    }
    return ProcessingController::kUnknownForward;
  }

  ProcessingController::ReverseState
  RootInputFileSequence::reverseState() const {
    if(rootFile_) {
      if(!rootFile_->wasFirstEventJustRead()) {
        return ProcessingController::kEventsBackwardsInFile;
      }
      if(fileIter_ != fileIterBegin_) {
        return ProcessingController::kPreviousFileExists;
      }
      return ProcessingController::kAtFirstEvent;
    }
    return ProcessingController::kUnknownReverse;
  }
}

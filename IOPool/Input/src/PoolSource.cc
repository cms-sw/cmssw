/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.69 2007/11/03 06:53:01 wmtan Exp $
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "RootFile.h"
#include "RootTree.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "FWCore/Catalog/interface/FileCatalog.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"
#include "TTree.h"
#include "TFile.h"

namespace edm {
  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    fileIter_(fileCatalogItems().begin()),
    rootFile_(),
    matchMode_(BranchDescription::Permissive),
    flatDistribution_(0),
    eventsRemainingInFile_(0),
    startAtRun_(pset.getUntrackedParameter<unsigned int>("firstRun", 1U)),
    startAtLumi_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1U)),
    startAtEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1U)),
    eventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0U)),
    forcedRunOffset_(0) {

    std::string matchMode = pset.getUntrackedParameter<std::string>("fileMatchMode", std::string("permissive"));
    if (matchMode == std::string("strict")) matchMode_ = BranchDescription::Strict;
    ClassFiller();
    if (primary()) {
      init(*fileIter_);
      forcedRunOffset_ = rootFile_->setForcedRunOffset(
			pset.getUntrackedParameter<unsigned int>("setRunNumber", 0));
      if (forcedRunOffset_ < 0) {
	RunNumber_t setRun = pset.getUntrackedParameter<unsigned int>("setRunNumber", 0);
        throw cms::Exception("Configuration")
          << "The value of the 'setRunNumber' parameter must not be\n"
	  << "less than the first run number in the first input file.\n"
          << "'setRunNumber' was " << setRun <<", while the first run was "
	  << setRun - forcedRunOffset_ << ".\n";
      }
      updateProductRegistry();
    } else {
      Service<RandomNumberGenerator> rng;
      if (!rng.isAvailable()) {
        throw cms::Exception("Configuration")
          << "A secondary input source requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
      }
      CLHEP::HepRandomEngine& engine = rng->getEngine();
      flatDistribution_ = new CLHEP::RandFlat(engine);
    }
  }

  void
  PoolSource::endJob() {
    rootFile_->close(true);
    delete flatDistribution_;
    flatDistribution_ = 0;
  }

  boost::shared_ptr<FileBlock>
  PoolSource::readFile_() {
    if (!initialized()) {
      // The first input file has already been opened.
      setInitialized();
    } else {
      // Open the next input file.
      if (!nextFile()) return boost::shared_ptr<FileBlock>();
    }
    return rootFile_->createFileBlock();
  }

  void PoolSource::init(FileCatalogItem const& file) {
    TTree::SetMaxTreeSize(kMaxLong64);
    rootFile_ = RootFileSharedPtr(new RootFile(file.fileName(), catalog().url(),
	processConfiguration(), file.logicalFileName(),
	startAtRun_, startAtLumi_, startAtEvent_, eventsToSkip_, remainingEvents(), forcedRunOffset_));
  }

  void PoolSource::updateProductRegistry() const {
    if (rootFile_->productRegistry()->nextID() > productRegistry()->nextID()) {
      productRegistryUpdate().setNextID(rootFile_->productRegistry()->nextID());
    }
    ProductRegistry::ProductList const& prodList = rootFile_->productRegistry()->productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
	it != itEnd; ++it) {
      productRegistryUpdate().copyProduct(it->second);
    }
  }

  bool PoolSource::nextFile() {
    // Account for events skipped in the file.
    eventsToSkip_ = rootFile_->eventsToSkip();

    if(fileIter_ != fileCatalogItems().end()) ++fileIter_;
    if(fileIter_ == fileCatalogItems().end()) {
      if (primary()) {
	return false;
      } else {
	fileIter_ = fileCatalogItems().begin();
      }
    }

    rootFile_->close(primary());

    init(*fileIter_);

    if (primary()) {
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    matchMode_);
      if (!mergeInfo.empty()) {
        throw cms::Exception("MismatchedInput","PoolSource::nextFile()") << mergeInfo;
      }
    }
    return true;
  }

  bool PoolSource::previousFile() {
    if(fileIter_ == fileCatalogItems().begin()) {
      if (primary()) {
	return false;
      } else {
	fileIter_ = fileCatalogItems().end();
      }
    }
    --fileIter_;

    rootFile_->close(primary());

    init(*fileIter_);

    if (primary()) {
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    matchMode_);
      if (!mergeInfo.empty()) {
        throw cms::Exception("MismatchedInput","PoolSource::previousEvent()") << mergeInfo;
      }
    }
    rootFile_->setToLastEntry();
    return true;
  }

  PoolSource::~PoolSource() {
  }

  boost::shared_ptr<RunPrincipal>
  PoolSource::readRun_() {
    return rootFile_->readRun(primary() ? productRegistry() : rootFile_->productRegistry()); 
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  PoolSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    return rootFile_->readLumi(primary() ? productRegistry() : rootFile_->productRegistry(), rp); 
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
  PoolSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    return rootFile_->readEvent(primary() ? productRegistry() : rootFile_->productRegistry(), lbp); 
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readAnEvent() {
    return rootFile_->readAnEvent(primary() ? productRegistry() : rootFile_->productRegistry()); 
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readIt(EventID const& id) {
    rootFile_->setCurrentPosition(id.run(), 0U, id.event());
    return readAnEvent();
  }

  // Rewind to before the first event that was read.
  void
  PoolSource::rewind_() {
    fileIter_ = fileCatalogItems().begin();
    init(*fileIter_);    
  }

  // Rewind to the beginning of the current file
  void
  PoolSource::rewindFile() {
    rootFile_->rewind();
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void
  PoolSource::skip(int offset) {
    while (offset != 0) {
      offset = rootFile_->skipEvents(offset);
      if (offset > 0 && !nextFile()) return;
      if (offset < 0 && !previousFile()) return;
    }
  }

  void
  PoolSource::readMany_(int number, EventPrincipalVector& result) {
    if (!primary()) {
	readRandom(number, result);
	return;
    }
    for (int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = readAnEvent();
      if (ev.get() == 0) {
	return;
      }
      EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
      --eventsRemainingInFile_;
    }
  }

  void
  PoolSource::readRandom(int number, EventPrincipalVector& result) {
    for (int i = 0; i < number; ++i) {
      while (eventsRemainingInFile_ <= 0) randomize();
      std::auto_ptr<EventPrincipal> ev = readAnEvent();
      if (ev.get() == 0) {
        rewindFile();
	ev = readAnEvent();
	assert(ev.get() != 0);
      }
      EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
      --eventsRemainingInFile_;
    }
  }

  void
  PoolSource::randomize() {
    FileCatalogItem const& file = *(fileCatalogItems().begin() +
				 flatDistribution_->fireInt(fileCatalogItems().size()));
    rootFile_ = RootFileSharedPtr(new RootFile(file.fileName(), catalog().url(),
        processConfiguration(), file.logicalFileName(), 0U, 0U, 0U, 0U, -1, 0));
    eventsRemainingInFile_ = rootFile_->eventTree().entries();
    rootFile_->setAtEventEntry(flatDistribution_->fireInt(eventsRemainingInFile_));
  }
}

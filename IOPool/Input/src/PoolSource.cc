/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.55 2007/06/28 23:10:16 wmtan Exp $
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "RootFile.h"
#include "RootTree.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "FWCore/Catalog/interface/FileCatalog.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"
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
    eventsRemainingInFile_(0)
  {
    std::string matchMode = pset.getUntrackedParameter<std::string>("fileMatchMode", std::string("permissive"));
    if (matchMode == std::string("strict")) matchMode_ = BranchDescription::Strict;
    ClassFiller();
    if (primary()) {
      init(*fileIter_);
      updateProductRegistry();
      setInitialPosition(pset);
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
    rootFile_->close();
    delete flatDistribution_;
  }

  void PoolSource::setInitialPosition(ParameterSet const& pset) {
    EventID firstEventID(pset.getUntrackedParameter<unsigned int>("firstRun", 0),
		  pset.getUntrackedParameter<unsigned int>("firstEvent", 0));
    if (firstEventID != EventID()) {
      EventID id = EventID(pset.getUntrackedParameter<unsigned int>("firstRun", 1),
		  pset.getUntrackedParameter<unsigned int>("firstEvent", 1));
      RootTree::EntryNumber eventEntry =
	rootFile_->eventTree().getBestEntryNumber(id.run(), id.event());
      while (eventEntry < 0) {
        // Set the entry to the last entry in this file
        rootFile_->eventTree().setEntryNumber(rootFile_->eventTree().entries()-1);

        // Advance to the first entry of the next file, if there is a next file.
        if(!next()) {
          throw cms::Exception("MismatchedInput","PoolSource::PoolSource()")
	    << "Input files have no " << id << "\n";
        }
        eventEntry = rootFile_->eventTree().getBestEntryNumber(id.run(), id.event());
      }
      RootTree::EntryNumber runEntry = rootFile_->runTree().getBestEntryNumber(id.run(), 0);
      RootTree::EntryNumber lumiEntry = rootFile_->lumiTree().getBestEntryNumber(id.run(), 1);
      if (runEntry < 0) runEntry = 0;
      if (lumiEntry < 0) lumiEntry = 0;
      rootFile_->eventTree().setEntryNumber(eventEntry - 1);
      rootFile_->lumiTree().setEntryNumber(lumiEntry - 1);
      rootFile_->runTree().setEntryNumber(runEntry - 1);
    }
    int eventsToSkip = pset.getUntrackedParameter<unsigned int>("skipEvents", 0);
    if (eventsToSkip > 0) {
      skip(eventsToSkip);
    }
    rootFile_->eventTree().setOrigEntryNumber();
    rootFile_->lumiTree().setOrigEntryNumber();
    rootFile_->runTree().setOrigEntryNumber();
  }

  void PoolSource::init(FileCatalogItem const& file) {
    TTree::SetMaxTreeSize(kMaxLong64);
    TFile *filePtr = (file.fileName().empty() ? 0 : TFile::Open(file.fileName().c_str()));
    if (filePtr != 0) filePtr->Close();
    rootFile_ = RootFileSharedPtr(new RootFile(file.fileName(), catalog().url(),
	processConfiguration(), file.logicalFileName()));
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

  bool PoolSource::next() {
    if (rootFile_->eventTree().next()) return true;
    if (!nextFile()) return false;
    return next();
  }

  bool PoolSource::nextFile() {
    if(fileIter_ != fileCatalogItems().end()) ++fileIter_;
    if(fileIter_ == fileCatalogItems().end()) {
      if (primary()) {
	return false;
      } else {
	fileIter_ = fileCatalogItems().begin();
      }
    }

    rootFile_->close();

    init(*fileIter_);

    if (primary()) {
      // make sure the new product registry is compatible with the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    matchMode_);
      if (!mergeInfo.empty()) {
        throw cms::Exception("MismatchedInput","PoolSource::next()") << mergeInfo;
      }
    }
    return true;
  }

  bool PoolSource::previous() {
    if(rootFile_->eventTree().previous()) return true;
    if(fileIter_ == fileCatalogItems().begin()) {
      if (primary()) {
	return false;
      } else {
	fileIter_ = fileCatalogItems().end();
      }
    }
    --fileIter_;

    rootFile_->close();

    init(*fileIter_);

    if (primary()) {
      // make sure the new product registry is compatible to the main one
      std::string mergeInfo = productRegistryUpdate().merge(*rootFile_->productRegistry(),
							    fileIter_->fileName(),
							    matchMode_);
      if (!mergeInfo.empty()) {
        throw cms::Exception("MismatchedInput","PoolSource::previous()") << mergeInfo;
      }
    }
    rootFile_->eventTree().setEntryNumber(rootFile_->eventTree().entries());
    return previous();
  }

  PoolSource::~PoolSource() {
  }

  boost::shared_ptr<RunPrincipal>
  PoolSource::readRun_() {
    boost::shared_ptr<RunPrincipal> rp;
    do {
	 rp = rootFile_->readRun(primary() ? productRegistry() : rootFile_->productRegistry()); 
    } while (rp.get() == 0 && nextFile());
    return rp;
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
  PoolSource::read() {
    if (!next()) {
      if (!primary()) {
	repeat();
      }
      return std::auto_ptr<EventPrincipal>(0);
    }
    boost::shared_ptr<LuminosityBlockPrincipal> lbp;
    return rootFile_->readEvent(primary() ? productRegistry() : rootFile_->productRegistry(), lbp); 
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readIt(EventID const& id) {
    RootTree::EntryNumber entry = rootFile_->eventTree().getBestEntryNumber(id.run(), id.event());
    if (entry >= 0) {
      rootFile_->eventTree().setEntryNumber(entry - 1);
      return read();
    } else {
      return std::auto_ptr<EventPrincipal>(0);
    }
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
    rootFile_->eventTree().resetEntryNumber();
    rootFile_->lumiTree().resetEntryNumber();
    rootFile_->runTree().resetEntryNumber();
  }

  // Advance "offset" events. Entry numbers begin at 0.
  // The current entry number is the last one read, not the next one read.
  // The current entry number may be -1, if none have been read yet.
  void
  PoolSource::skip(int offset) {
    EntryNumber newEntry = rootFile_->eventTree().entryNumber() + offset;
    if (newEntry >= rootFile_->eventTree().entries()) {

      // We must go to the next file
      // Calculate how much we will advance in this file,
      // including one for the next() call below
      int increment = rootFile_->eventTree().entries() - rootFile_->eventTree().entryNumber();    

      // Set the entry to the last entry in this file
      rootFile_->eventTree().setEntryNumber(rootFile_->eventTree().entries()-1);

      // Advance to the first entry of the next file, if there is a next file.
      if(!next()) return;

      // Now skip the remaining offset.
      skip(offset - increment);

    } else if (newEntry < -1) {

      // We must go to the previous file
      // Calculate how much we will back up in this file,
      // including one for the previous() call below
      int decrement = rootFile_->eventTree().entryNumber() + 1;    

      // Set the entry to the first entry in this file
      rootFile_->eventTree().setEntryNumber(0);

      // Back up to the last entry of the previous file, if there is a previous file.
      if(!previous()) return;

      // Now skip the remaining offset.
      skip(offset + decrement);
    } else {
      // The same file.
      rootFile_->eventTree().setEntryNumber(newEntry);
    }
  }

  void
  PoolSource::readMany_(int number, EventPrincipalVector& result) {
    if (!primary()) {
	readRandom(number, result);
	return;
    }
    for (int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = read();
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
      std::auto_ptr<EventPrincipal> ev = read();
      if (ev.get() == 0) {
	rewindFile();
	ev = read();
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
        processConfiguration(), file.logicalFileName()));
    eventsRemainingInFile_ = rootFile_->eventTree().entries();
    rootFile_->eventTree().setEntryNumber(flatDistribution_->fireInt(eventsRemainingInFile_));
  }
}

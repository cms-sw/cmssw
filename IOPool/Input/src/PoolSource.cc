/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.38 2006/10/24 20:29:02 wmtan Exp $
----------------------------------------------------------------------*/
#include "IOPool/Input/src/PoolSource.h"
#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Input/src/RootTree.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TTree.h"

namespace edm {
  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    fileIter_(fileCatalogItems().begin()),
    rootFile_(),
    origRootFile_(),
    matchMode_(BranchDescription::Permissive)
  {
    std::string matchMode = pset.getUntrackedParameter<std::string>("fileMatchMode", std::string("permissive"));
    if (matchMode == std::string("strict")) matchMode_ = BranchDescription::Strict;
    ClassFiller();
    init(*fileIter_);
    if (primary()) {
      updateProductRegistry();
    }
    setInitialPosition(pset);
  }

  void
  PoolSource::endJob() {
    rootFile_->close();
  }

  void PoolSource::setInitialPosition(ParameterSet const& pset) {
    EventID firstEventID(pset.getUntrackedParameter<unsigned int>("firstRun", 0),
		  pset.getUntrackedParameter<unsigned int>("firstEvent", 0));
    if (firstEventID != EventID()) {
      EventID id = EventID(pset.getUntrackedParameter<unsigned int>("firstRun", 1),
		  pset.getUntrackedParameter<unsigned int>("firstEvent", 1));
      RootTree::EntryNumber entry = rootFile_->eventTree().getBestEntryNumber(id.run(), id.event());
      while (entry < 0) {
        // Set the entry to the last entry in this file
        rootFile_->eventTree().setEntryNumber(rootFile_->eventTree().entries()-1);

        // Advance to the first entry of the next file, if there is a next file.
        if(!next()) {
          throw cms::Exception("MismatchedInput","PoolSource::PoolSource()")
	    << "Input files have no " << id << "\n";
        }
        entry = rootFile_->eventTree().getBestEntryNumber(id.run(), id.event());
      }
      rootFile_->eventTree().setEntryNumber(entry - 1);
    }
    int eventsToSkip = pset.getUntrackedParameter<unsigned int>("skipEvents", 0);
    if (eventsToSkip > 0) {
      skip(eventsToSkip);
    }
    origRootFile_ = rootFile_;
    rootFile_->eventTree().setOrigEntryNumber();
    rootFile_->lumiTree().setOrigEntryNumber();
    rootFile_->runTree().setOrigEntryNumber();
  }

  void PoolSource::init(FileCatalogItem const& file) {
    TTree::SetMaxTreeSize(kMaxLong64);
    rootFile_ = RootFileSharedPtr(new RootFile(file.fileName(), catalog().url(), file.logicalFileName()));
  }

  void PoolSource::updateProductRegistry() const {
    if (rootFile_->productRegistry().nextID() > productRegistry().nextID()) {
      productRegistry().setNextID(rootFile_->productRegistry().nextID());
    }
    ProductRegistry::ProductList const& prodList = rootFile_->productRegistry().productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
	it != prodList.end(); ++it) {
      productRegistry().copyProduct(it->second);
    }
  }

  bool PoolSource::next() {
    if(rootFile_->eventTree().next()) return true;
    ++fileIter_;
    if(fileIter_ == fileCatalogItems().end()) {
      if (primary()) {
	return false;
      } else {
	fileIter_ = fileCatalogItems().begin();
      }
    }

    // save the product registry from the current file, temporarily
    boost::shared_ptr<ProductRegistry> pReg(rootFile_->productRegistrySharedPtr());

    rootFile_->close();

    init(*fileIter_);

    ProductRegistry * preg = (primary() ? &productRegistry() : pReg.get());

    // make sure the new product registry is compatible with the main one
    std::string mergeInfo = preg->merge(rootFile_->productRegistry(), fileIter_->fileName(), matchMode_);
    if (!mergeInfo.empty()) {
      throw cms::Exception("MismatchedInput","PoolSource::next()")
        << mergeInfo;
    }
    return next();
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

    // save the product registry from the current file, temporarily
    boost::shared_ptr<ProductRegistry> pReg(rootFile_->productRegistrySharedPtr());

    rootFile_->close();

    init(*fileIter_);

    ProductRegistry * preg = (primary() ? &productRegistry() : pReg.get());

    // make sure the new product registry is compatible to the main one
    std::string mergeInfo = preg->merge(rootFile_->productRegistry(), fileIter_->fileName(), matchMode_);
    if (!mergeInfo.empty()) {
      throw cms::Exception("MismatchedInput","PoolSource::previous()")
        << mergeInfo;
    }
    rootFile_->eventTree().setEntryNumber(rootFile_->eventTree().entries());
    return previous();
  }

  PoolSource::~PoolSource() {
  }

  // read() is responsible for creating, and setting up, the
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
  PoolSource::read() {
    if (!next()) {
      if (!primary()) {
	repeat();
      }
      return std::auto_ptr<EventPrincipal>(0);
    }
    return rootFile_->read(primary() ? productRegistry() : rootFile_->productRegistry()); 
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
    rootFile_ = origRootFile_;
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
    for (int i = 0; i < number; ++i) {
      std::auto_ptr<EventPrincipal> ev = read();
      if (ev.get() == 0) {
	return;
      }
      EventPrincipalVectorElement e(ev.release());
      result.push_back(e);
    }
  }
}

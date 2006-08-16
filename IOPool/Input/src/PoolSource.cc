/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.33 2006/08/01 05:39:24 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/PoolSource.h"
#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    fileIter_(fileNames().begin()),
    rootFile_(),
    origRootFile_(),
    origEntryNumber_(),
    matchMode_(BranchDescription::Permissive),
    mainInput_(pset.getParameter<std::string>("@module_label") == std::string("@main_input"))
  {
    std::string matchMode = pset.getUntrackedParameter<std::string>("fileMatchMode", std::string("permissive"));
    if (matchMode == std::string("strict")) matchMode_ = BranchDescription::Strict;
    ClassFiller();
    init(*fileIter_);
    if (mainInput_) {
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
      RootFile::EntryNumber entry = rootFile_->getEntryNumber(id);
      while (entry < 0) {
        // Set the entry to the last entry in this file
        rootFile_->setEntryNumber(rootFile_->entries()-1);

        // Advance to the first entry of the next file, if there is a next file.
        if(!next()) {
          throw cms::Exception("MismatchedInput","PoolSource::PoolSource()")
	    << "Input files have no " << id << "\n";
        }
        entry = rootFile_->getEntryNumber(id);
      }
      rootFile_->setEntryNumber(entry - 1);
    }
    int eventsToSkip = pset.getUntrackedParameter<unsigned int>("skipEvents", 0);
    if (eventsToSkip > 0) {
      skip(eventsToSkip);
    }
    origRootFile_ = rootFile_;
    origEntryNumber_ = rootFile_->entryNumber();
  }

  void PoolSource::init(std::string const& file) {

    rootFile_ = RootFileSharedPtr(new RootFile(file, catalog().url()));
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
    if(rootFile_->next()) return true;
    ++fileIter_;
    if(fileIter_ == fileNames().end()) {
      if (mainInput_) {
	return false;
      } else {
	fileIter_ = fileNames().begin();
      }
    }

    // save the product registry from the current file, temporarily
    boost::shared_ptr<ProductRegistry> pReg(rootFile_->productRegistrySharedPtr());

    rootFile_->close();

    init(*fileIter_);

    ProductRegistry * preg = (mainInput_ ? &productRegistry() : pReg.get());

    // make sure the new product registry is compatible with the main one
    if (!preg->merge(rootFile_->productRegistry(), matchMode_)) {
      throw cms::Exception("MismatchedInput","PoolSource::next()")
	<< "File " << *fileIter_ << "\nhas different product registry than previous files\n";
    }
    return next();
  }

  bool PoolSource::previous() {
    if(rootFile_->previous()) return true;
    if(fileIter_ == fileNames().begin()) {
      if (mainInput_) {
	return false;
      } else {
	fileIter_ = fileNames().end();
      }
    }
    --fileIter_;

    // save the product registry from the current file, temporarily
    boost::shared_ptr<ProductRegistry> pReg(rootFile_->productRegistrySharedPtr());

    rootFile_->close();

    init(*fileIter_);

    ProductRegistry * preg = (mainInput_ ? &productRegistry() : pReg.get());

    // make sure the new product registry is compatible to the main one
    if (!preg->merge(rootFile_->productRegistry(), matchMode_)) {
      throw cms::Exception("MismatchedInput","PoolSource::previous()")
	<< "File " << *fileIter_ << "\nhas different product registry than previous files\n";
    }
    rootFile_->setEntryNumber(rootFile_->entries());
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
      if (!mainInput_) {
	repeat();
      }
      return std::auto_ptr<EventPrincipal>(0);
    }
    return rootFile_->read(mainInput_ ? productRegistry() : rootFile_->productRegistry()); 
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readIt(EventID const& id) {
    RootFile::EntryNumber entry = rootFile_->getEntryNumber(id);
    if (entry >= 0) {
      rootFile_->setEntryNumber(entry - 1);
      return read();
    } else {
      return std::auto_ptr<EventPrincipal>(0);
    }
  }

  // Rewind to before the first event that was read.
  void
  PoolSource::rewind_() {
    rootFile_ = origRootFile_;
    rootFile_->setEntryNumber(origEntryNumber_);
  }

  // Advance "offset" events. Entry numbers begin at 0.
  // The current entry number is the last one read, not the next one read.
  // The current entry number may be -1, if none have been read yet.
  void
  PoolSource::skip(int offset) {
    EntryNumber newEntry = rootFile_->entryNumber() + offset;
    if (newEntry >= rootFile_->entries()) {

      // We must go to the next file
      // Calculate how much we will advance in this file,
      // including one for the next() call below
      int increment = rootFile_->entries() - rootFile_->entryNumber();    

      // Set the entry to the last entry in this file
      rootFile_->setEntryNumber(rootFile_->entries()-1);

      // Advance to the first entry of the next file, if there is a next file.
      if(!next()) return;

      // Now skip the remaining offset.
      skip(offset - increment);

    } else if (newEntry < -1) {

      // We must go to the previous file
      // Calculate how much we will back up in this file,
      // including one for the previous() call below
      int decrement = rootFile_->entryNumber() + 1;    

      // Set the entry to the first entry in this file
      rootFile_->setEntryNumber(0);

      // Back up to the last entry of the previous file, if there is a previous file.
      if(!previous()) return;

      // Now skip the remaining offset.
      skip(offset + decrement);
    } else {
      // The same file.
      rootFile_->setEntryNumber(newEntry);
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

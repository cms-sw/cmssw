/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.14 2006/01/06 02:38:07 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/PoolSource.h"
#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  PoolRASource::PoolRASource(ParameterSet const& pset, InputSourceDescription const& desc) :
    InputSource(desc),
    catalog_(PoolCatalog::READ,
      PoolCatalog::toPhysical(pset.getUntrackedParameter("catalog", std::string()))),
    file_(pset.getUntrackedParameter("fileName", std::string())),
    files_(pset.getUntrackedParameter("fileNames", std::vector<std::string>())),
    fileIter_(files_.begin()),
    rootFile_(),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)) {
    ClassFiller();
    if (file_.empty()) {
      if (files_.empty()) { // this will throw;
        pset.getUntrackedParameter<std::string>("fileName");
      } else {
        init(*fileIter_);
        ++fileIter_;
      }
    } else {
      init(file_);
    }
    updateRegistry();
  }

  void PoolRASource::init(std::string const& file) {

    // delete the old RootFile, if any.  The file will be closed.
    rootFile_.reset();
    std::string pfn;
    catalog_.findFile(pfn, file);

    rootFile_ = boost::shared_ptr<RootFile>(new RootFile(pfn));
  }

  void PoolRASource::updateRegistry() const {
    if (rootFile_->productRegistry().nextID() > productRegistry().nextID()) {
      productRegistry().setNextID(rootFile_->productRegistry().nextID());
    }
    ProductRegistry::ProductList const& prodList = rootFile_->productRegistry().productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      productRegistry().copyProduct(it->second);
    }
  }

  bool PoolRASource::next() {
    if(rootFile_->next()) return true;
    if(fileIter_ == files_.end()) return false;

    // save the product registry from the current file, temporarily
    boost::shared_ptr<ProductRegistry const> pReg(rootFile_->productRegistrySharedPtr());

    init(*fileIter_);

    // make sure the new product registry is identical to the old one
    if (*pReg != rootFile_->productRegistry()) {
      throw cms::Exception("MismatchedInput","PoolSource::next()")
        << "File " << *fileIter_ << "\nhas different product registry than previous files\n";
    }
    ++fileIter_;
    return next();
  }

  PoolRASource::~PoolRASource() {
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
  PoolRASource::read() {
    // If we're done, or out of range, return a null auto_ptr
    if (remainingEvents_ == 0) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    if (!next()) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    --remainingEvents_;
    return rootFile_->read(productRegistry()); 
  }

  std::auto_ptr<EventPrincipal>
  PoolRASource::read(EventID const& id) {
    // For now, don't support multiple runs.
    assert (id.run() == rootFile_->eventID().run());
    // For now, assume EventID's are all there.
    int offset = static_cast<long>(id.event()) - static_cast<long>(rootFile_->eventID().event());
    rootFile_->entryNumber() += offset;
    return read();
  }

  void
  PoolRASource::skip(int offset) {
    rootFile_->entryNumber() += offset;
  }
}

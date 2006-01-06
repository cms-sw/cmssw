/*----------------------------------------------------------------------
$Id: PoolSecondarySource.cc,v 1.12 2005/12/02 22:40:22 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/SecondaryInput/src/PoolSecondarySource.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/EDProduct/interface/ProductID.h"

namespace edm {
  PoolSecondarySource::PoolSecondarySource(ParameterSet const& pset) :
    SecondaryInputSource(),
    catalog_(PoolCatalog::READ,
      PoolCatalog::toPhysical(pset.getUntrackedParameter("catalog", std::string()))),
    productMap_(),
    files_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames")),
    fileIter_(files_.begin()),
    rootFile_() {
    ClassFiller();
    init(*fileIter_);
    ++fileIter_;
  }

  void PoolSecondarySource::init(std::string const& file) {
    productMap_.clear();
    std::string pfn;
    catalog_.findFile(pfn, file);

    rootFile_ = boost::shared_ptr<RootFile>(new RootFile(pfn));
    ProductRegistry::ProductList const& prodList = rootFile_->productRegistry().productList();

    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      productMap_.insert(std::make_pair(it->second.productID_, it->second));
    }
  }

  bool PoolSecondarySource::next() {
    if(rootFile_->next()) return true;
    if(fileIter_ == files_.end()) fileIter_ = files_.begin();

    // delete the old RootFile.  The file will be closed.
    rootFile_.reset();

    init(*fileIter_);

    ++fileIter_;
    return next();
  }


  PoolSecondarySource::~PoolSecondarySource() {
  }

  // read_() is responsible for creating, and setting up, the
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
  void
  PoolSecondarySource::read_(int idx, int number, std::vector<EventPrincipal*>& result) {
    
    for (int entry = idx, i = 0; i < number; ++entry, ++i) {
      next();
      std::auto_ptr<EventPrincipal> ev = rootFile_->read(rootFile_->productRegistry(), productMap_);
      result.push_back(ev.release());
    }
  }
}

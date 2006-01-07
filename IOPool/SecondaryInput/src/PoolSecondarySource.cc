/*----------------------------------------------------------------------
$Id: PoolSecondarySource.cc,v 1.13 2006/01/06 02:39:17 wmtan Exp $
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
    file_(pset.getUntrackedParameter("fileName", std::string())),
    files_(pset.getUntrackedParameter("fileNames", std::vector<std::string>())),
    fileIter_(files_.begin()),
    rootFile_() {
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
  }

  void PoolSecondarySource::init(std::string const& file) {
    // delete the old RootFile, if any.  The file will be closed.
    rootFile_.reset();
    std::string pfn;
    catalog_.findFile(pfn, file);

    rootFile_ = boost::shared_ptr<RootFile>(new RootFile(pfn));
  }

  bool PoolSecondarySource::next() {
    if(rootFile_->next()) return true;
    if (files_.empty()) {
      rootFile_.reset();
      init(file_);
      return next();
    }
    if(fileIter_ == files_.end()) fileIter_ = files_.begin();

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
      std::auto_ptr<EventPrincipal> ev = rootFile_->read(rootFile_->productRegistry());
      result.push_back(ev.release());
    }
  }
}

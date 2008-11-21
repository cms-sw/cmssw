/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Principal.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  DataViewImpl::DataViewImpl(Principal & pcpl,
	ModuleDescription const& md,
	BranchType const& branchType)  :
    putProducts_(),
    principal_(pcpl),
    md_(md),
    branchType_(branchType)
  {  }

  struct deleter {
    void operator()(std::pair<EDProduct*, ConstBranchDescription const*> const p) const { delete p.first; }
  };

  DataViewImpl::~DataViewImpl() {
    // anything left here must be the result of a failure
    // let's record them as failed attempts in the event principal
    for_all(putProducts_, deleter());
  }

  size_t
  DataViewImpl::size() const {
    return putProducts_.size() + principal_.size();
  }

  BasicHandle
  DataViewImpl::get_(TypeID const& tid, SelectorBase const& sel) const
  {
    return principal_.getBySelector(tid, sel);
  }

  void
  DataViewImpl::getMany_(TypeID const& tid,
		  SelectorBase const& sel,
		  BasicHandleVec& results) const
  {
    principal_.getMany(tid, sel, results);
  }

  BasicHandle
  DataViewImpl::getByLabel_(TypeID const& tid,
                     std::string const& label,
  	             std::string const& productInstanceName,
  	             std::string const& processName) const
  {
    return principal_.getByLabel(tid, label, productInstanceName, processName);
  }

  BasicHandle
  DataViewImpl::getByType_(TypeID const& tid) const
  {
    return principal_.getByType(tid);
  }

  void
  DataViewImpl::getManyByType_(TypeID const& tid,
		  BasicHandleVec& results) const
  {
    principal_.getManyByType(tid, results);
  }

  int
  DataViewImpl::getMatchingSequence_(TypeID const& typeID,
                                     SelectorBase const& selector,
                                     BasicHandleVec& results,
                                     bool stopIfProcessHasMatch) const
  {
    return principal_.getMatchingSequence(typeID,
                                    selector,
                                    results,
                                    stopIfProcessHasMatch);
  }

  int
  DataViewImpl::getMatchingSequenceByLabel_(TypeID const& typeID,
                                            std::string const& label,
                                            std::string const& productInstanceName,
                                            BasicHandleVec& results,
                                            bool stopIfProcessHasMatch) const
  {
    edm::Selector sel(edm::ModuleLabelSelector(label) &&
                      edm::ProductInstanceNameSelector(productInstanceName));

    int n = principal_.getMatchingSequence(typeID,
                                     sel,
                                     results,
                                     stopIfProcessHasMatch);
    return n;
  }

  int
  DataViewImpl::getMatchingSequenceByLabel_(TypeID const& typeID,
                                            std::string const& label,
                                            std::string const& productInstanceName,
                                            std::string const& processName,
                                            BasicHandleVec& results,
                                            bool stopIfProcessHasMatch) const
  {
    edm::Selector sel(edm::ModuleLabelSelector(label) &&
                      edm::ProductInstanceNameSelector(productInstanceName) &&
                      edm::ProcessNameSelector(processName) );

    int n = principal_.getMatchingSequence(typeID,
  				   sel,
  				   results,
  				   stopIfProcessHasMatch);
    return n;
  }

  ProcessHistory const&
  DataViewImpl::processHistory() const
  {
    return principal_.processHistory();
  }

  ConstBranchDescription const&
  DataViewImpl::getBranchDescription(TypeID const& type,
				     std::string const& productInstanceName) const {
    std::string friendlyClassName = type.friendlyClassName();
        BranchKey bk(friendlyClassName, md_.moduleLabel(), productInstanceName, md_.processName());
    ProductRegistry::ConstProductList const& pl = principal_.productRegistry().constProductList();
    ProductRegistry::ConstProductList::const_iterator it = pl.find(bk);
    if (it == pl.end()) {
      throw edm::Exception(edm::errors::InsertFailure)
	<< "Illegal attempt to 'put' an unregistered product.\n"
	<< "No product is registered for\n"
	<< "  process name:                '" << bk.processName_ << "'\n"
	<< "  module label:                '" << bk.moduleLabel_ << "'\n"
	<< "  product friendly class name: '" << bk.friendlyClassName_ << "'\n"
	<< "  product instance name:       '" << bk.productInstanceName_ << "'\n"

	<< "The ProductRegistry contains:\n"
	<< principal_.productRegistry()
	<< '\n';
    }
    if(it->second.branchType() != branchType_) {
        throw edm::Exception(edm::errors::InsertFailure,"Not Registered")
          << "put: Problem found while adding product. "
          << "The product for ("
          << bk.friendlyClassName_ << ","
          << bk.moduleLabel_ << ","
          << bk.productInstanceName_ << ","
          << bk.processName_
          << ")\n"
          << "is registered for a(n) " << it->second.branchType()
          << " instead of for a(n) " << branchType_
          << ".\n";
    }
    return it->second;
  }

  EDProductGetter const*
  DataViewImpl::prodGetter() const{
    return principal_.prodGetter();
  }
}

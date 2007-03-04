/*----------------------------------------------------------------------
$Id: DataViewImpl.cc,v 1.11 2007/02/17 23:31:30 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DataBlockImpl.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Group.h"

using namespace std;

namespace edm {

  DataViewImpl::DataViewImpl(DataBlockImpl & dbk,
	ModuleDescription const& md,
	BranchType const& branchType)  :
    put_products_(),
    gotProductIDs_(),
    dbk_(dbk),
    md_(md),
    branchType_(branchType)
  {  }

  struct deleter {
    void operator()(pair<EDProduct*, BranchDescription const*> const p) const { delete p.first; }
  };

  DataViewImpl::~DataViewImpl() {
    // anything left here must be the result of a failure
    // let's record them as failed attempts in the event principal
    for_each(put_products_.begin(),put_products_.end(),deleter());
  }

  size_t
  DataViewImpl::size() const {
    return put_products_.size() + dbk_.size();
  }

  void 
  DataViewImpl::commit_() {
    // fill in guts of provenance here
    ProductPtrVec::iterator pit(put_products_.begin());
    ProductPtrVec::iterator pie(put_products_.end());

    while(pit!=pie) {
	auto_ptr<EDProduct> pr(pit->first);
	// note: ownership has been passed - so clear the pointer!
	pit->first = 0;

	auto_ptr<Provenance> pv(new Provenance(*pit->second, BranchEntryDescription::Success));

	// set parts of provenance
	pv->event.cid_ = 0; // TODO: what is this supposed to be?
	pv->event.isPresent_ = true;
	pv->event.parents_ = gotProductIDs_;
	pv->event.moduleDescriptionID_ = pit->second->moduleDescriptionID_;

	dbk_.put(pr,pv);
	++pit;
    }

    // the cleanup is all or none
    put_products_.clear();
  }

  BasicHandle
  DataViewImpl::get_(ProductID const& oid) const
  {
    return dbk_.get(oid);
  }

  BasicHandle
  DataViewImpl::get_(TypeID const& tid, SelectorBase const& sel) const
  {
    return dbk_.getBySelector(tid, sel);
  }
    
  BasicHandle
  DataViewImpl::getByLabel_(TypeID const& tid,
		     string const& label,
                     string const & productInstanceName) const
  {
    return dbk_.getByLabel(tid, label, productInstanceName);
  }

  void 
  DataViewImpl::getMany_(TypeID const& tid, 
		  SelectorBase const& sel,
		  BasicHandleVec& results) const
  {
    dbk_.getMany(tid, sel, results);
  }

  BasicHandle
  DataViewImpl::getByLabel_(TypeID const& tid,
                     string const& label,
  	             string const& productInstanceName,
  	             string const& processName) const
  {
    return dbk_.getByLabel(tid, label, productInstanceName, processName);
  }

  BasicHandle
  DataViewImpl::getByType_(TypeID const& tid) const
  {
    return dbk_.getByType(tid);
  }

  void 
  DataViewImpl::getManyByType_(TypeID const& tid, 
		  BasicHandleVec& results) const
  {
    dbk_.getManyByType(tid, results);
  }

  BasicHandle
  DataViewImpl::getMatchingSequence_(type_info const& wantedElementType,
				     string const& moduleLabel,
				     string const& productInstanceName) const
  {
    return dbk_.getMatchingSequence(wantedElementType,
				    moduleLabel,
				    productInstanceName);
  }

  Provenance const&
  DataViewImpl::getProvenance(ProductID const& oid) const
  {
    return dbk_.getProvenance(oid);
  }

  void
  DataViewImpl::getAllProvenance(vector<Provenance const*> & provenances) const
  {
    dbk_.getAllProvenance(provenances);
  }

  ProcessHistory const&
  DataViewImpl::processHistory() const
  {
    return dbk_.processHistory();
  }

  BranchDescription const&
  DataViewImpl::getBranchDescription(TypeID const& type,
				     string const& productInstanceName) const {
    string friendlyClassName = type.friendlyClassName();
        BranchKey bk(friendlyClassName, md_.moduleLabel(), productInstanceName, md_.processName());
    ProductRegistry::ProductList const& pl = dbk_.productRegistry().productList();
    ProductRegistry::ProductList::const_iterator it = pl.find(bk);
    if (it == pl.end()) {
      throw edm::Exception(edm::errors::InsertFailure)
	<< "Illegal attempt to 'put' an unregistered product.\n"
	<< "No product is registered for\n"
	<< "  process name:                '" << bk.processName_ << "'\n"
	<< "  module label:                '" << bk.moduleLabel_ << "'\n"
	<< "  product friendly class name: '" << bk.friendlyClassName_ << "'\n"
	<< "  product instance name:       '" << bk.productInstanceName_ << "'\n"

	<< "The ProductRegistry contains:\n"
	<< dbk_.productRegistry()
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
    return dbk_.prodGetter();
  }
}

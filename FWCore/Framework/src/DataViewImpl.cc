/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Principal.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Framework/interface/Selector.h"

namespace edm {

  DataViewImpl::DataViewImpl(Principal & pcpl,
	ModuleDescription const& md)  :
    putProducts_(),
    principal_(pcpl),
    md_(md) {
  }

  DataViewImpl::~DataViewImpl() {
  }

  BranchType const&
  DataViewImpl::branchType() const {
    return principal_.branchType();
  }

  size_t
  DataViewImpl::size() const {
    return putProducts_.size() + principal_.size();
  }

  BasicHandle
  DataViewImpl::get_(TypeID const& tid, SelectorBase const& sel) const {
    return principal_.getBySelector(tid, sel);
  }

  void
  DataViewImpl::getMany_(TypeID const& tid,
		  SelectorBase const& sel,
		  BasicHandleVec& results) const {
    principal_.getMany(tid, sel, results);
  }

  BasicHandle
  DataViewImpl::getByLabel_(TypeID const& tid,
                     std::string const& label,
  	             std::string const& productInstanceName,
  	             std::string const& processName) const {
    size_t cachedOffset = 0;
    int fillCount = -1;
    return principal_.getByLabel(tid, label, productInstanceName, processName, cachedOffset, fillCount);
  }

  BasicHandle
  DataViewImpl::getByLabel_(TypeID const& tid,
                     InputTag const& tag) const {

    principal_.maybeFlushCache(tid, tag);
    return principal_.getByLabel(tid, tag.label(), tag.instance(), tag.process(), tag.cachedOffset(), tag.fillCount());
  }

  BasicHandle
  DataViewImpl::getByType_(TypeID const& tid) const {
    return principal_.getByType(tid);
  }

  void
  DataViewImpl::getManyByType_(TypeID const& tid,
		  BasicHandleVec& results) const {
    principal_.getManyByType(tid, results);
  }

  int
  DataViewImpl::getMatchingSequence_(TypeID const& typeID,
                                     SelectorBase const& selector,
                                     BasicHandle& result) const {
    return principal_.getMatchingSequence(typeID,
                                    selector,
                                    result);
  }

  int
  DataViewImpl::getMatchingSequenceByLabel_(TypeID const& typeID,
                                            std::string const& label,
                                            std::string const& productInstanceName,
                                            BasicHandle& result) const {
    Selector sel(ModuleLabelSelector(label) &&
                 ProductInstanceNameSelector(productInstanceName));

    int n = principal_.getMatchingSequence(typeID,
                                     sel,
                                     result);
    return n;
  }

  int
  DataViewImpl::getMatchingSequenceByLabel_(TypeID const& typeID,
                                            std::string const& label,
                                            std::string const& productInstanceName,
                                            std::string const& processName,
                                            BasicHandle& result) const {
    Selector sel(ModuleLabelSelector(label) &&
                 ProductInstanceNameSelector(productInstanceName) &&
                 ProcessNameSelector(processName) );

    int n = principal_.getMatchingSequence(typeID,
  				   sel,
  				   result);
    return n;
  }

  ProcessHistory const&
  DataViewImpl::processHistory() const {
    return principal_.processHistory();
  }

  ConstBranchDescription const&
  DataViewImpl::getBranchDescription(TypeID const& type,
				     std::string const& productInstanceName) const {
    TransientProductLookupMap const& tplm = principal_.productRegistry().productLookup();
    std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> range = 
     tplm.equal_range(TypeInBranchType(type,branchType()),md_.moduleLabel(),productInstanceName);
   
    //NOTE: getBranchDescription should only be called by a EDProducer and therefore the processName should
    // match the first one returned by equal_range since they are ordered by time. However, there is one test
    // which violates this rule (FWCore/Framework/test/Event_t.cpp.  I do not see a go way to 'fix' it so
    // I'll allow the same behavior it depends upon
    bool foundMatch = false;
    if(range.first != range.second) {
       foundMatch = true;
       while(md_.processName() != range.first->branchDescription()->processName()) {
          ++range.first;
          if(range.first == range.second || range.first->isFirst()) {
             foundMatch = false;
             break;
          }
       }
    }
    if(!foundMatch) {
      throw edm::Exception(edm::errors::InsertFailure)
	<< "Illegal attempt to 'put' an unregistered product.\n"
	<< "No product is registered for\n"
	<< "  process name:                '" << md_.processName() << "'\n"
	<< "  module label:                '" << md_.moduleLabel() << "'\n"
	<< "  product friendly class name: '" << type.friendlyClassName() << "'\n"
	<< "  product instance name:       '" << productInstanceName << "'\n"

	<< "The ProductRegistry contains:\n"
	<< principal_.productRegistry()
	<< '\n';
    }
    return *(range.first->branchDescription());
  }

  EDProductGetter const*
  DataViewImpl::prodGetter() const{
    return principal_.prodGetter();
  }
}

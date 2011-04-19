/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>

#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Selector.h"

namespace edm {

  PrincipalGetAdapter::PrincipalGetAdapter(Principal & pcpl,
	ModuleDescription const& md)  :
    //putProducts_(),
    principal_(pcpl),
    md_(md) {
  }

  PrincipalGetAdapter::~PrincipalGetAdapter() {
  }


  void
  principal_get_adapter_detail::deleter::operator()(std::pair<WrapperHolder, ConstBranchDescription const*> const p) const {
    WrapperHolder* edp = const_cast<WrapperHolder*>(&p.first);
    edp->reset();
  }

  void
  principal_get_adapter_detail::throwOnPutOfNullProduct(
	char const* principalType,
	TypeID const& productType,
	std::string const& productInstanceName) {
      throw Exception(errors::NullPointerError)
	<< principalType
	<< "::put: A null auto_ptr was passed to 'put'.\n"
	<< "The pointer is of type "
	<< productType
        << ".\nThe specified productInstanceName was '"
	<< productInstanceName
        << "'.\n";
  }

  BranchType const&
  PrincipalGetAdapter::branchType() const {
    return principal_.branchType();
  }

  BasicHandle
  PrincipalGetAdapter::get_(TypeID const& tid, SelectorBase const& sel) const {
    return principal_.getBySelector(tid, sel);
  }

  void
  PrincipalGetAdapter::getMany_(TypeID const& tid,
		  SelectorBase const& sel,
		  BasicHandleVec& results) const {
    principal_.getMany(tid, sel, results);
  }

  BasicHandle
  PrincipalGetAdapter::getByLabel_(TypeID const& tid,
                     std::string const& label,
  	             std::string const& productInstanceName,
  	             std::string const& processName) const {
    size_t cachedOffset = 0;
    int fillCount = -1;
    return principal_.getByLabel(tid, label, productInstanceName, processName, cachedOffset, fillCount);
  }

  BasicHandle
  PrincipalGetAdapter::getByLabel_(TypeID const& tid,
                     InputTag const& tag) const {

    principal_.maybeFlushCache(tid, tag);
    return principal_.getByLabel(tid, tag.label(), tag.instance(), tag.process(), tag.cachedOffset(), tag.fillCount());
  }

  BasicHandle
  PrincipalGetAdapter::getByType_(TypeID const& tid) const {
    return principal_.getByType(tid);
  }

  void
  PrincipalGetAdapter::getManyByType_(TypeID const& tid,
		  BasicHandleVec& results) const {
    principal_.getManyByType(tid, results);
  }

  int
  PrincipalGetAdapter::getMatchingSequence_(TypeID const& typeID,
                                     SelectorBase const& selector,
                                     BasicHandle& result) const {
    return principal_.getMatchingSequence(typeID,
                                    selector,
                                    result);
  }

  int
  PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
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
  PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                            std::string const& label,
                                            std::string const& productInstanceName,
                                            std::string const& processName,
                                            BasicHandle& result) const {
    Selector sel(ModuleLabelSelector(label) &&
                 ProductInstanceNameSelector(productInstanceName) &&
                 ProcessNameSelector(processName));

    int n = principal_.getMatchingSequence(typeID,
  				   sel,
  				   result);
    return n;
  }

  ProcessHistory const&
  PrincipalGetAdapter::processHistory() const {
    return principal_.processHistory();
  }

  ConstBranchDescription const&
  PrincipalGetAdapter::getBranchDescription(TypeID const& type,
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
  PrincipalGetAdapter::prodGetter() const{
    return principal_.prodGetter();
  }
}

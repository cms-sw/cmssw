/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>

#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProductHolderIndexHelper.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  principal_get_adapter_detail::deleter::operator()(std::pair<WrapperOwningHolder, ConstBranchDescription const*> const p) const {
    WrapperOwningHolder* edp = const_cast<WrapperOwningHolder*>(&p.first);
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

  void
  principal_get_adapter_detail::throwOnPrematureRead(
	char const* principalType,
	TypeID const& productType,
	std::string const& moduleLabel,
	std::string const& productInstanceName) {
      //throw Exception(errors::LogicError)
      LogWarning("LogicError")
	<< "::getByLabel: An attempt was made to read a "
	<< principalType
        << " product before end"
	<< principalType
        << "() was called.\n"
	<< "The product is of type '"
	<< productType
        << "'.\nThe specified ModuleLabel was '"
	<< moduleLabel
        << "'.\nThe specified productInstanceName was '"
	<< productInstanceName
        << "'.\n";
  }

  void
  principal_get_adapter_detail::throwOnPrematureRead(
	char const* principalType,
	TypeID const& productType) {
      //throw Exception(errors::LogicError)
      LogWarning("LogicError")
	<< "::getManyByType: An attempt was made to read a "
	<< principalType
        << " product before end"
	<< principalType
        << "() was called.\n"
	<< "The product is of type '"
	<< productType
        << "'.\n";
  }
  
  void
  principal_get_adapter_detail::throwOnPrematureRead(
                                                     char const* principalType,
                                                     TypeID const& productType,
                                                     EDGetToken token) {
    throw Exception(errors::LogicError)
    << "::getByToken: An attempt was made to read a "
    << principalType
    << " product before end"
    << principalType
    << "() was called.\n"
    << "The index of the token was "<<token.value()<<".\n";
  }

  BasicHandle
  PrincipalGetAdapter::makeFailToGetException(KindOfType kindOfType,
                                              TypeID const& productType,
                                              EDGetToken token) const {
    EDConsumerBase::Labels labels;
    consumer_->labelsForToken(token,labels);
    boost::shared_ptr<cms::Exception> exception(new Exception(errors::ProductNotFound));
    if (kindOfType == PRODUCT_TYPE) {
      *exception << "Principal::getByToken: Found zero products matching all criteria\nLooking for type: " << productType << "\n"
      << "Looking for module label: " << labels.module << "\n" << "Looking for productInstanceName: " << labels.productInstance << "\n"
      << (0==labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n";
    } else {
      *exception << "Principal::getByToken: Found zero products matching all criteria\nLooking for a container with elements of type: " << productType << "\n"
      << "Looking for module label: " << labels.module << "\n" << "Looking for productInstanceName: " << labels.productInstance << "\n"
      << (0==labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n";
    }
    return BasicHandle(exception);
  }


  BranchType const&
  PrincipalGetAdapter::branchType() const {
    return principal_.branchType();
  }

  BasicHandle
  PrincipalGetAdapter::getByLabel_(TypeID const& typeID,
                                   InputTag const& tag) const {
    return principal_.getByLabel(PRODUCT_TYPE, typeID, tag);
  }

  BasicHandle
  PrincipalGetAdapter::getByLabel_(TypeID const& typeID,
                                   std::string const& label,
  	                           std::string const& instance,
  	                           std::string const& process) const {
    return principal_.getByLabel(PRODUCT_TYPE, typeID, label, instance, process);
  }
  
  BasicHandle
  PrincipalGetAdapter::getByToken_(TypeID const& id, KindOfType kindOfType, EDGetToken token) const {
    ProductHolderIndex index = consumer_->indexFrom(token,InEvent,id);
    if( unlikely(index == ProductHolderIndexInvalid)) {
      return makeFailToGetException(kindOfType,id,token);
    }
    return principal_.getByToken(kindOfType,id,index);
  }
  

  BasicHandle
  PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                                   InputTag const& tag) const {
    return principal_.getByLabel(ELEMENT_TYPE, typeID, tag);    
  }

  BasicHandle
  PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                                   std::string const& label,
                                                   std::string const& instance,
                                                   std::string const& process) const {
    return principal_.getByLabel(ELEMENT_TYPE,
                                 typeID,
                                 label,
                                 instance,
                                 process);
  }

  void
  PrincipalGetAdapter::getManyByType_(TypeID const& tid,
		  BasicHandleVec& results) const {
    principal_.getManyByType(tid, results);
  }

  ProcessHistory const&
  PrincipalGetAdapter::processHistory() const {
    return principal_.processHistory();
  }

  ConstBranchDescription const&
  PrincipalGetAdapter::getBranchDescription(TypeID const& type,
				     std::string const& productInstanceName) const {
    ProductHolderIndexHelper const& productHolderIndexHelper = principal_.productLookup();
    ProductHolderIndex index = productHolderIndexHelper.index(PRODUCT_TYPE, type, md_.moduleLabel().c_str(),productInstanceName.c_str(), md_.processName().c_str());
    if(index == ProductHolderIndexInvalid) {
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
    ProductHolderBase const*  phb = principal_.getProductByIndex(index, false, false);
    assert(phb != nullptr);
    return phb->branchDescription();
  }

  EDProductGetter const*
  PrincipalGetAdapter::prodGetter() const{
    return principal_.prodGetter();
  }

  bool
  PrincipalGetAdapter::isComplete() const {
    return principal_.isComplete();
  }
}

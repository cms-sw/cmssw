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
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {

  PrincipalGetAdapter::PrincipalGetAdapter(Principal & pcpl,
	ModuleDescription const& md)  :
    //putProducts_(),
    principal_(pcpl),
    md_(md),
    consumer_(nullptr)
  {
  }

  PrincipalGetAdapter::~PrincipalGetAdapter() {
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
    << "The index of the token was "<<token.index()<<".\n";
  }
  
  BasicHandle
  PrincipalGetAdapter::makeFailToGetException(KindOfType kindOfType,
                                              TypeID const& productType,
                                              EDGetToken token) const {
    EDConsumerBase::Labels labels;
    consumer_->labelsForToken(token,labels);
    //no need to copy memory since the exception will no occur after the
    // const char* have been deleted
    return BasicHandle(makeHandleExceptionFactory([labels,kindOfType,productType]()->std::shared_ptr<cms::Exception> {
      std::shared_ptr<cms::Exception> exception(std::make_shared<Exception>(errors::ProductNotFound));
      if (kindOfType == PRODUCT_TYPE) {
        *exception << "Principal::getByToken: Found zero products matching all criteria\nLooking for type: " << productType << "\n"
        << "Looking for module label: " << labels.module << "\n" << "Looking for productInstanceName: " << labels.productInstance << "\n"
        << (0==labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n";
      } else {
        *exception << "Principal::getByToken: Found zero products matching all criteria\nLooking for a container with elements of type: " << productType << "\n"
        << "Looking for module label: " << labels.module << "\n" << "Looking for productInstanceName: " << labels.productInstance << "\n"
        << (0==labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n";
      }
      return exception;
    }));
  }

  void
  PrincipalGetAdapter::throwAmbiguousException(TypeID const& productType,
                                               EDGetToken token) const {
    EDConsumerBase::Labels labels;
    consumer_->labelsForToken(token,labels);
    cms::Exception exception("AmbiguousProduct");
    exception << "Principal::getByToken: More than 1 product matches all criteria\nLooking for a container with elements of type: " << productType << "\n"
    << "Looking for module label: " << labels.module << "\n" << "Looking for productInstanceName: " << labels.productInstance << "\n"
    << (0==labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n"
    << "This can only occur with get function calls using a Handle<View> argument.\n"
    << "Try a get not using a View or change the instance name of one of the products";
    throw exception;    
  }

  BranchType const&
  PrincipalGetAdapter::branchType() const {
    return principal_.branchType();
  }

  BasicHandle
  PrincipalGetAdapter::getByLabel_(TypeID const& typeID,
                                   InputTag const& tag,
                                   ModuleCallingContext const* mcc) const {
    return principal_.getByLabel(PRODUCT_TYPE, typeID, tag, consumer_, mcc);
  }

  BasicHandle
  PrincipalGetAdapter::getByLabel_(TypeID const& typeID,
                                   std::string const& label,
  	                           std::string const& instance,
  	                           std::string const& process,
                                   ModuleCallingContext const* mcc) const {
    return principal_.getByLabel(PRODUCT_TYPE, typeID, label, instance, process, consumer_, mcc);
  }
  
  BasicHandle
  PrincipalGetAdapter::getByToken_(TypeID const& id, KindOfType kindOfType, EDGetToken token,
                                   ModuleCallingContext const* mcc) const {
    ProductHolderIndexAndSkipBit indexAndBit = consumer_->indexFrom(token,branchType(),id);
    ProductHolderIndex index = indexAndBit.productHolderIndex();
    bool skipCurrentProcess = indexAndBit.skipCurrentProcess();
    if( unlikely(index == ProductHolderIndexInvalid)) {
      return makeFailToGetException(kindOfType,id,token);
    } else if( unlikely(index == ProductHolderIndexAmbiguous)) {
      // This deals with ambiguities where the process is specified
      throwAmbiguousException(id, token);
    }
    bool ambiguous = false;
    BasicHandle h = principal_.getByToken(kindOfType, id, index, skipCurrentProcess, ambiguous, mcc);
    if (ambiguous) {
      // This deals with ambiguities where the process is not specified
      throwAmbiguousException(id, token);
    } else if(!h.isValid()) {
      return makeFailToGetException(kindOfType,id,token);
    }
    return h;
  }

  BasicHandle
  PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                                   InputTag const& tag,
                                                   ModuleCallingContext const* mcc) const {
    return principal_.getByLabel(ELEMENT_TYPE, typeID, tag, consumer_, mcc);
  }

  BasicHandle
  PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                                   std::string const& label,
                                                   std::string const& instance,
                                                   std::string const& process,
                                                   ModuleCallingContext const* mcc) const {
    auto h= principal_.getByLabel(ELEMENT_TYPE,
                                  typeID,
                                  label,
                                  instance,
                                  process,
                                  consumer_,
                                  mcc);
    return h;
  }

  void
  PrincipalGetAdapter::getManyByType_(TypeID const& tid,
                                      BasicHandleVec& results,
                                      ModuleCallingContext const* mcc) const {
    principal_.getManyByType(tid, results, consumer_, mcc);
  }

  ProcessHistory const&
  PrincipalGetAdapter::processHistory() const {
    return principal_.processHistory();
  }

  BranchDescription const&
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
    ProductHolderBase const*  phb = principal_.getProductHolderByIndex(index);
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

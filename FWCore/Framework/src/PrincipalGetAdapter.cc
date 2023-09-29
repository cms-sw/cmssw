/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>

#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ProducerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {

  PrincipalGetAdapter::PrincipalGetAdapter(Principal const& pcpl, ModuleDescription const& md, bool isComplete)
      :  //putProducts_(),
        principal_(pcpl),
        md_(md),
        consumer_(nullptr),
        resourcesAcquirer_(nullptr),
        isComplete_(isComplete) {}

  PrincipalGetAdapter::~PrincipalGetAdapter() {}

  void principal_get_adapter_detail::throwOnPutOfNullProduct(char const* principalType,
                                                             TypeID const& productType,
                                                             std::string const& productInstanceName) {
    throw Exception(errors::NullPointerError)
        << principalType << "::put: A null unique_ptr was passed to 'put'.\n"
        << "The pointer is of type " << productType << ".\nThe specified productInstanceName was '"
        << productInstanceName << "'.\n";
  }

  void principal_get_adapter_detail::throwOnPutOfUninitializedToken(char const* principalType,
                                                                    std::type_info const& type) {
    TypeID productType{type};
    throw Exception(errors::LogicError) << principalType << "::put: An uninitialized EDPutToken was passed to 'put'.\n"
                                        << "The pointer is of type " << productType << ".\n";
  }

  void principal_get_adapter_detail::throwOnPutOfWrongType(std::type_info const& wrongType, TypeID const& rightType) {
    TypeID wrongTypeID{wrongType};
    throw Exception(errors::LogicError) << "The registered type for an EDPutToken does not match the put type.\n"
                                        << "The expected type " << rightType << "\nThe put type " << wrongTypeID
                                        << ".\n";
  }

  void principal_get_adapter_detail::throwOnPrematureRead(char const* principalType,
                                                          TypeID const& productType,
                                                          std::string const& moduleLabel,
                                                          std::string const& productInstanceName) {
    //throw Exception(errors::LogicError)
    LogWarning("LogicError") << "::getByLabel: An attempt was made to read a " << principalType << " product before end"
                             << principalType << "() was called.\n"
                             << "The product is of type '" << productType << "'.\nThe specified ModuleLabel was '"
                             << moduleLabel << "'.\nThe specified productInstanceName was '" << productInstanceName
                             << "'.\n";
  }

  void principal_get_adapter_detail::throwOnPrematureRead(char const* principalType,
                                                          TypeID const& productType,
                                                          EDGetToken token) {
    throw Exception(errors::LogicError) << "::getByToken: An attempt was made to read a " << principalType
                                        << " product before end" << principalType << "() was called.\n"
                                        << "The index of the token was " << token.index() << ".\n";
  }

  size_t PrincipalGetAdapter::numberOfProductsConsumed() const { return consumer_->itemsToGetFrom(InEvent).size(); }

  void PrincipalGetAdapter::labelsForToken(EDGetToken const& iToken, ProductLabels& oLabels) const {
    consumer_->labelsForToken(iToken, oLabels);
  }

  unsigned int PrincipalGetAdapter::processBlockIndex(std::string const& processName) const {
    return principal_.processBlockIndex(processName);
  }

  BasicHandle PrincipalGetAdapter::makeFailToGetException(KindOfType kindOfType,
                                                          TypeID const& productType,
                                                          EDGetToken token) const {
    EDConsumerBase::Labels labels;
    consumer_->labelsForToken(token, labels);
    //no need to copy memory since the exception will no occur after the
    // const char* have been deleted
    return BasicHandle(
        makeHandleExceptionFactory([labels, kindOfType, productType]() -> std::shared_ptr<cms::Exception> {
          std::shared_ptr<cms::Exception> exception(std::make_shared<Exception>(errors::ProductNotFound));
          if (kindOfType == PRODUCT_TYPE) {
            *exception << "Principal::getByToken: Found zero products matching all criteria\nLooking for type: "
                       << productType << "\n"
                       << "Looking for module label: " << labels.module << "\n"
                       << "Looking for productInstanceName: " << labels.productInstance << "\n"
                       << (0 == labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n";
          } else {
            *exception << "Principal::getByToken: Found zero products matching all criteria\nLooking for a container "
                          "with elements of type: "
                       << productType << "\n"
                       << "Looking for module label: " << labels.module << "\n"
                       << "Looking for productInstanceName: " << labels.productInstance << "\n"
                       << (0 == labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n";
          }
          return exception;
        }));
  }

  void PrincipalGetAdapter::throwAmbiguousException(TypeID const& productType, EDGetToken token) const {
    EDConsumerBase::Labels labels;
    consumer_->labelsForToken(token, labels);
    cms::Exception exception("AmbiguousProduct");
    exception << "Principal::getByToken: More than 1 product matches all criteria\nLooking for a container with "
                 "elements of type: "
              << productType << "\n"
              << "Looking for module label: " << labels.module << "\n"
              << "Looking for productInstanceName: " << labels.productInstance << "\n"
              << (0 == labels.process[0] ? "" : "Looking for process: ") << labels.process << "\n"
              << "This can only occur with get function calls using a Handle<View> argument.\n"
              << "Try a get not using a View or change the instance name of one of the products";
    throw exception;
  }

  BranchType const& PrincipalGetAdapter::branchType() const { return principal_.branchType(); }

  BasicHandle PrincipalGetAdapter::getByLabel_(TypeID const& typeID,
                                               InputTag const& tag,
                                               ModuleCallingContext const* mcc) const {
    return principal_.getByLabel(PRODUCT_TYPE, typeID, tag, consumer_, resourcesAcquirer_, mcc);
  }

  BasicHandle PrincipalGetAdapter::getByLabel_(TypeID const& typeID,
                                               std::string const& label,
                                               std::string const& instance,
                                               std::string const& process,
                                               ModuleCallingContext const* mcc) const {
    return principal_.getByLabel(PRODUCT_TYPE, typeID, label, instance, process, consumer_, resourcesAcquirer_, mcc);
  }

  BasicHandle PrincipalGetAdapter::getByToken_(TypeID const& id,
                                               KindOfType kindOfType,
                                               EDGetToken token,
                                               ModuleCallingContext const* mcc) const {
    ProductResolverIndexAndSkipBit indexAndBit = consumer_->indexFrom(token, branchType(), id);
    ProductResolverIndex index = indexAndBit.productResolverIndex();
    bool skipCurrentProcess = indexAndBit.skipCurrentProcess();
    if (UNLIKELY(index == ProductResolverIndexInvalid)) {
      return makeFailToGetException(kindOfType, id, token);
    } else if (UNLIKELY(index == ProductResolverIndexAmbiguous)) {
      // This deals with ambiguities where the process is specified
      throwAmbiguousException(id, token);
    }
    bool ambiguous = false;
    BasicHandle h =
        principal_.getByToken(kindOfType, id, index, skipCurrentProcess, ambiguous, resourcesAcquirer_, mcc);
    if (ambiguous) {
      // This deals with ambiguities where the process is not specified
      throwAmbiguousException(id, token);
    } else if (!h.isValid()) {
      return makeFailToGetException(kindOfType, id, token);
    }
    return h;
  }

  BasicHandle PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                                               InputTag const& tag,
                                                               ModuleCallingContext const* mcc) const {
    return principal_.getByLabel(ELEMENT_TYPE, typeID, tag, consumer_, resourcesAcquirer_, mcc);
  }

  BasicHandle PrincipalGetAdapter::getMatchingSequenceByLabel_(TypeID const& typeID,
                                                               std::string const& label,
                                                               std::string const& instance,
                                                               std::string const& process,
                                                               ModuleCallingContext const* mcc) const {
    auto h = principal_.getByLabel(ELEMENT_TYPE, typeID, label, instance, process, consumer_, resourcesAcquirer_, mcc);
    return h;
  }

  ProcessHistory const& PrincipalGetAdapter::processHistory() const { return principal_.processHistory(); }

  void PrincipalGetAdapter::throwUnregisteredPutException(TypeID const& type,
                                                          std::string const& productInstanceName) const {
    std::ostringstream str;
    for (auto branchDescription : principal_.productRegistry().allBranchDescriptions()) {
      if (branchDescription->moduleLabel() == md_.moduleLabel() and
          branchDescription->processName() == md_.processName()) {
        str << *branchDescription << "-----\n";
      }
    }
    throw edm::Exception(edm::errors::InsertFailure)
        << "Illegal attempt to 'put' an unregistered product.\n"
        << "No product is registered for\n"
        << "  product friendly class name: '" << type.friendlyClassName() << "'\n"
        << "  module label:                '" << md_.moduleLabel() << "'\n"
        << "  product instance name:       '" << productInstanceName << "'\n"
        << "  process name:                '" << md_.processName() << "'\n"

        << "The following data products are registered for production by " << md_.moduleLabel() << ":\n"
        << str.str() << '\n'
        << "To correct the problem:\n"
           "   1) make sure the proper 'produce' call is being made in the module's constructor,\n"
           "   2) if 'produce' exists and uses a product instance name make sure that same name is used during the "
           "'put' call.";
  }

  BranchDescription const& PrincipalGetAdapter::getBranchDescription(TypeID const& type,
                                                                     std::string const& productInstanceName) const {
    ProductResolverIndexHelper const& productResolverIndexHelper = principal_.productLookup();
    ProductResolverIndex index = productResolverIndexHelper.index(
        PRODUCT_TYPE, type, md_.moduleLabel().c_str(), productInstanceName.c_str(), md_.processName().c_str());
    if (UNLIKELY(index == ProductResolverIndexInvalid)) {
      throwUnregisteredPutException(type, productInstanceName);
    }
    ProductResolverBase const* phb = principal_.getProductResolverByIndex(index);
    assert(phb != nullptr);
    return phb->branchDescription();
  }

  BranchDescription const& PrincipalGetAdapter::getBranchDescription(unsigned int iPutTokenIndex) const {
    auto index = prodBase_->putTokenIndexToProductResolverIndex()[iPutTokenIndex];
    ProductResolverBase const* phb = principal_.getProductResolverByIndex(index);
    assert(phb != nullptr);
    return phb->branchDescription();
  }

  ProductID const& PrincipalGetAdapter::getProductID(unsigned int iPutTokenIndex) const {
    auto index = prodBase_->putTokenIndexToProductResolverIndex()[iPutTokenIndex];
    ProductResolverBase const* phb = principal_.getProductResolverByIndex(index);
    assert(phb != nullptr);
    auto prov = phb->stableProvenance();
    assert(prov != nullptr);
    return prov->productID();
  }

  Transition PrincipalGetAdapter::transition() const {
    if (LIKELY(principal().branchType() == InEvent)) {
      return Transition::Event;
    }
    if (principal().branchType() == InRun) {
      if (isComplete()) {
        return Transition::EndRun;
      } else {
        return Transition::BeginRun;
      }
    }
    if (isComplete()) {
      return Transition::EndLuminosityBlock;
    }
    return Transition::BeginLuminosityBlock;
    //Must be lumi
  }

  EDPutToken::value_type PrincipalGetAdapter::getPutTokenIndex(TypeID const& type,
                                                               std::string const& productInstanceName) const {
    auto tran = transition();
    size_t index = 0;
    for (auto const& tl : prodBase_->typeLabelList()) {
      if ((tran == tl.transition_) and (type == tl.typeID_) and (productInstanceName == tl.productInstanceName_)) {
        return index;
      }
      ++index;
    }
    throwUnregisteredPutException(type, productInstanceName);
    return std::numeric_limits<unsigned int>::max();
  }

  std::string const& PrincipalGetAdapter::productInstanceLabel(EDPutToken iToken) const {
    return prodBase_->typeLabelList()[iToken.index()].productInstanceName_;
  }

  TypeID const& PrincipalGetAdapter::getTypeIDForPutTokenIndex(EDPutToken::value_type index) const {
    return prodBase_->typeLabelList()[index].typeID_;
  }

  std::vector<edm::ProductResolverIndex> const& PrincipalGetAdapter::putTokenIndexToProductResolverIndex() const {
    return prodBase_->putTokenIndexToProductResolverIndex();
  }

  std::vector<bool> const& PrincipalGetAdapter::recordProvenanceList() const {
    return prodBase_->recordProvenanceList();
  }

  EDProductGetter const* PrincipalGetAdapter::prodGetter() const { return principal_.prodGetter(); }
}  // namespace edm

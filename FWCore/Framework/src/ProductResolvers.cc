/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "ProductResolvers.h"
#include "Worker.h"
#include "UnscheduledAuxiliary.h"
#include "UnscheduledConfigurator.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>


namespace edm {

  void DataManagingProductResolver::throwProductDeletedException() const {
    ProductDeletedException exception;
    exception << "ProductResolverBase::resolveProduct_: The product matching all criteria was already deleted\n"
    << "Looking for type: " << branchDescription().unwrappedTypeID() << "\n"
    << "Looking for module label: " << moduleLabel() << "\n"
    << "Looking for productInstanceName: " << productInstanceName() << "\n"
    << (processName().empty() ? "" : "Looking for process: ") << processName() << "\n"
    << "This means there is a configuration error.\n"
    << "The module which is asking for this data must be configured to state that it will read this data.";
    throw exception;
    
  }

  //This is a templated function in order to avoid calling another virtual function
  template <bool callResolver, typename FUNC>
  ProductData const*
  DataManagingProductResolver::resolveProductImpl(FUNC resolver, ResolveStatus& resolveStatus) const {
    
    if(productWasDeleted()) {
      throwProductDeletedException();
    }
    auto presentStatus = status();
    
    if(callResolver && presentStatus == ProductStatus::ResolveNotRun) {
      //if resolver fails because of exception or not setting product
      // make sure the status goes to failed
      auto failedStatusSetter = [this](ProductStatus* presentStatus) {
        if(this->status() == ProductStatus::ResolveNotRun) {
          this->setFailedStatus();
        }
        *presentStatus = this->status();
      };
      std::unique_ptr<ProductStatus, decltype(failedStatusSetter)> failedStatusGuard(&presentStatus, failedStatusSetter);

      //If successful, this will call setProduct
      resolver();
    }
    
    
    if (presentStatus == ProductStatus::ProductSet) {
      auto pd = &getProductData();
      if(pd->wrapper()->isPresent()) {
        resolveStatus = ProductFound;
        return pd;
      }
    }

    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void
  DataManagingProductResolver::mergeProduct(std::unique_ptr<WrapperBase> iFrom) const {
    assert(status() == ProductStatus::ProductSet);
    if(not iFrom) { return;}
    
    checkType(*iFrom);
    
    auto original =getProductData().unsafe_wrapper();
    if(original->isMergeable()) {
      original->mergeProduct(iFrom.get());
    } else if(original->hasIsProductEqual()) {
      if(!original->isProductEqual(iFrom.get())) {
        auto const& bd = branchDescription();
        edm::LogError("RunLumiMerging")
        << "ProductResolver::mergeTheProduct\n"
        << "Two run/lumi products for the same run/lumi which should be equal are not\n"
        << "Using the first, ignoring the second\n"
        << "className = " << bd.className() << "\n"
        << "moduleLabel = " << bd.moduleLabel() << "\n"
        << "instance = " << bd.productInstanceName() << "\n"
        << "process = " << bd.processName() << "\n";
      }
    } else {
      auto const& bd = branchDescription();
      edm::LogWarning("RunLumiMerging")
      << "ProductResolver::mergeTheProduct\n"
      << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
      << "Using the first, ignoring the second in merge\n"
      << "className = " << bd.className() << "\n"
      << "moduleLabel = " << bd.moduleLabel() << "\n"
      << "instance = " << bd.productInstanceName() << "\n"
      << "process = " << bd.processName() << "\n";
    }
  }
  

  ProductData const*
  InputProductResolver::resolveProduct_(ResolveStatus& resolveStatus,
                                      Principal const& principal,
                                      bool,
                                      SharedResourcesAcquirer* ,
                                      ModuleCallingContext const* mcc) const {
    return resolveProductImpl<true>([this,&principal,mcc]() {
                                return principal.readFromSource(*this, mcc); }
                              ,resolveStatus);
                              
  }

  ProductData const*
  PuttableProductResolver::resolveProduct_(ResolveStatus& resolveStatus,
                                          Principal const&,
                                          bool skipCurrentProcess,
                                          SharedResourcesAcquirer*,
                                          ModuleCallingContext const*) const {
    if (!skipCurrentProcess) {
      //'false' means never call the lambda function
      return resolveProductImpl<false>([](){return;}, resolveStatus);
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void
  UnscheduledProductResolver::setupUnscheduled(UnscheduledConfigurator const& iConfigure) {
    aux_ = iConfigure.auxiliary();
    worker_ = iConfigure.findWorker(branchDescription().moduleLabel());
    assert(worker_ != nullptr);
    
  }  
  
  ProductData const*
  UnscheduledProductResolver::resolveProduct_(ResolveStatus& resolveStatus,
                                            Principal const& principal,
                                            bool skipCurrentProcess,
                                            SharedResourcesAcquirer* sra,
                                            ModuleCallingContext const* mcc) const {
    if (!skipCurrentProcess and worker_) {
      return resolveProductImpl<true>(
        [&principal,this,sra,mcc]() {
          try {
            auto const& event = static_cast<EventPrincipal const&>(principal);
            ParentContext parentContext(mcc);
            aux_->preModuleDelayedGetSignal_.emit(*(mcc->getStreamContext()),*mcc);
            std::shared_ptr<void> guard(nullptr,[this,mcc](const void*){
              aux_->postModuleDelayedGetSignal_.emit(*(mcc->getStreamContext()),*mcc);
            });

            auto workCall = [this,&event,&parentContext,mcc] () {worker_->doWork<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> >(
                                           event,
                                           *(aux_->eventSetup()),
                                           event.streamID(),
                                           parentContext,
                                           mcc->getStreamContext());
            };
            
            if (sra) {
              sra->temporaryUnlock(workCall);
            } else {
              workCall();
            }

          }
          catch (cms::Exception & ex) {
            std::ostringstream ost;
            ost << "Calling produce method for unscheduled module "
            << worker_->description().moduleName() << "/'"
            << worker_->description().moduleLabel() << "'";
            ex.addContext(ost.str());
            throw;
          }
        },
        resolveStatus);
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void
  ProducedProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if(status() != defaultStatus()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << branchDescription().branchName() << "\n";
    }
    assert(edp.get() != nullptr);
    
    setProduct(std::move(edp));  // ProductResolver takes ownership
  }

  void
  InputProductResolver::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    setProduct(std::move(edp));
  }

  
  void
  DataManagingProductResolver::connectTo(ProductResolverBase const& iOther, Principal const*) {
    assert(false);
  }
  
  void
  DataManagingProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> prod) const {
    if(not prod) {return;}
    if(status() == defaultStatus()) {
      //resolveProduct has not been called or it failed
      putProduct(std::move(prod));
    } else {
      mergeProduct(std::move(prod));
    }
  }
  

  
  void
  DataManagingProductResolver::checkType(WrapperBase const& prod) const {
    // Check if the types match.
    TypeID typeID(prod.dynamicTypeInfo());
    if(typeID != branchDescription().unwrappedTypeID()) {
      // Types do not match.
      throw Exception(errors::EventCorruption)
      << "Product on branch " << branchDescription().branchName() << " is of wrong type.\n"
      << "It is supposed to be of type " << branchDescription().className() << ".\n"
      << "It is actually of type " << typeID.className() << ".\n";
    }
  }
  

  void
  DataManagingProductResolver::setProduct(std::unique_ptr<WrapperBase> edp) const {
    if(edp) {
      checkType(*edp);
      productData_.unsafe_setWrapper(std::move(edp));
      theStatus_ = ProductStatus::ProductSet;
    } else {
      setFailedStatus();
    }
  }
  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  DataManagingProductResolver::productUnavailable_() const {
    auto presentStatus = status();
    if(presentStatus == ProductStatus::ProductSet) {
      return !(getProductData().wrapper()->isPresent());
    }
    return presentStatus != ProductStatus::ResolveNotRun;
  }
    
  bool
  DataManagingProductResolver::productResolved_() const {
    auto s = status();
    return (s != defaultStatus() ) or (s == ProductStatus::ProductDeleted);
  }

  
  // This routine returns true if the product was deleted early in order to save memory
  bool
  DataManagingProductResolver::productWasDeleted_() const {
    return status() == ProductStatus::ProductDeleted;
  }
  
  void DataManagingProductResolver::setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    productData_.setProvenance(provRetriever,ph,pid);
  }
  
  void DataManagingProductResolver::setProcessHistory_(ProcessHistory const& ph) {
    productData_.setProcessHistory(ph);
  }
  
  ProductProvenance const* DataManagingProductResolver::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }
  
  void DataManagingProductResolver::resetProductData_(bool deleteEarly) {
    if(theStatus_ == ProductStatus::ProductSet) {
      productData_.resetProductData();
    }
    if(deleteEarly) {
      theStatus_ = ProductStatus::ProductDeleted;
    } else {
      resetStatus();
    }
  }
  
  bool DataManagingProductResolver::singleProduct_() const {
    return true;
  }

  NoProcessProductResolver::
  NoProcessProductResolver(std::vector<ProductResolverIndex> const&  matchingHolders,
                         std::vector<bool> const& ambiguous) :
    matchingHolders_(matchingHolders),
    ambiguous_(ambiguous) {
    assert(ambiguous_.size() == matchingHolders_.size());
  }

  ProductData const* NoProcessProductResolver::resolveProduct_(ResolveStatus& resolveStatus,
                                                             Principal const& principal,
                                                             bool skipCurrentProcess,
                                                             SharedResourcesAcquirer* sra,
                                                             ModuleCallingContext const* mcc) const {
    std::vector<unsigned int> const& lookupProcessOrder = principal.lookupProcessOrder();
    for(unsigned int k : lookupProcessOrder) {
      assert(k < ambiguous_.size());
      if(k == 0) break; // Done
      if(ambiguous_[k]) {
        resolveStatus = Ambiguous;
        return nullptr;
      }
      if (matchingHolders_[k] != ProductResolverIndexInvalid) {
        ProductResolverBase const* productResolver = principal.getProductResolverByIndex(matchingHolders_[k]);
        ProductData const* pd =  productResolver->resolveProduct(resolveStatus, principal, skipCurrentProcess, sra, mcc);
        if(pd != nullptr) return pd;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void AliasProductResolver::setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    realProduct_.setProvenance(provRetriever,ph,pid);
  }

  void AliasProductResolver::setProcessHistory_(ProcessHistory const& ph) {
    realProduct_.setProcessHistory(ph);
  }

  ProductProvenance const* AliasProductResolver::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void AliasProductResolver::resetProductData_(bool deleteEarly) {
    realProduct_.resetProductData_(deleteEarly);
  }

  bool AliasProductResolver::singleProduct_() const {
    return true;
  }
  
  void AliasProductResolver::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
    << "AliasProductResolver::putProduct_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  
  void AliasProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
    << "AliasProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  

  
  void ParentProcessProductResolver::setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    provRetriever_ = provRetriever;
  }
  
  void ParentProcessProductResolver::setProcessHistory_(ProcessHistory const& ph) {
  }
  
  ProductProvenance const* ParentProcessProductResolver::productProvenancePtr_() const {
    return provRetriever_? provRetriever_->branchIDToProvenance(bd_->branchID()): nullptr;
  }
  
  void ParentProcessProductResolver::resetProductData_(bool deleteEarly) {
  }
  
  bool ParentProcessProductResolver::singleProduct_() const {
    return true;
  }

  void ParentProcessProductResolver::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
    << "ParentProcessProductResolver::putProduct_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  
  void ParentProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
    << "ParentProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  
  void NoProcessProductResolver::setProvenance_(ProductProvenanceRetriever const* , ProcessHistory const& , ProductID const& ) {
  }

  void NoProcessProductResolver::setProcessHistory_(ProcessHistory const& ) {
  }

  ProductProvenance const* NoProcessProductResolver::productProvenancePtr_() const {
    return nullptr;
  }

  void NoProcessProductResolver::resetProductData_(bool) {
  }

  bool NoProcessProductResolver::singleProduct_() const {
    return false;
  }

  bool NoProcessProductResolver::unscheduledWasNotRun_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::unscheduledWasNotRun_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productUnavailable_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::productUnavailable_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productResolved_() const {
    throw Exception(errors::LogicError)
    << "NoProcessProductResolver::productResolved_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }

  bool NoProcessProductResolver::productWasDeleted_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::productWasDeleted_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::putProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  BranchDescription const& NoProcessProductResolver::branchDescription_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::branchDescription_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::resetBranchDescription_(std::shared_ptr<BranchDescription const>) {
    throw Exception(errors::LogicError)
      << "NoProcessProductResolver::resetBranchDescription_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  Provenance const* NoProcessProductResolver::provenance_() const {
    throw Exception(errors::LogicError)
    << "NoProcessProductResolver::provenance_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }

  void NoProcessProductResolver::connectTo(ProductResolverBase const&, Principal const*) {
    throw Exception(errors::LogicError)
    << "NoProcessProductResolver::connectTo() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
    
  }
}

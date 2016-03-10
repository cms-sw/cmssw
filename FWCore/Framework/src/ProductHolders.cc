/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "ProductHolders.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>
namespace edm {
  
  //This is a templated function in order to avoid calling another virtual function
  template <bool callResolver, typename FUNC>
  ProductData const*
  DataManagingProductHolder::resolveProductImpl(FUNC resolver, ResolveStatus& resolveStatus) const {
    
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
  DataManagingProductHolder::mergeProduct(std::unique_ptr<WrapperBase> iFrom) const {
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
        << "ProductHolder::mergeTheProduct\n"
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
      << "ProductHolder::mergeTheProduct\n"
      << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
      << "Using the first, ignoring the second in merge\n"
      << "className = " << bd.className() << "\n"
      << "moduleLabel = " << bd.moduleLabel() << "\n"
      << "instance = " << bd.productInstanceName() << "\n"
      << "process = " << bd.processName() << "\n";
    }
  }
  

  ProductData const*
  InputProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
                                      Principal const& principal,
                                      bool,
                                      SharedResourcesAcquirer* ,
                                      ModuleCallingContext const* mcc) const {
    return resolveProductImpl<true>([this,&principal,mcc]() {
                                return principal.readFromSource(*this, mcc); }
                              ,resolveStatus);
                              
  }

  ProductData const*
  PuttableProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
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

  ProductData const*
  UnscheduledProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
                                            Principal const& principal,
                                            bool skipCurrentProcess,
                                            SharedResourcesAcquirer* sra,
                                            ModuleCallingContext const* mcc) const {
    if (!skipCurrentProcess) {
      return resolveProductImpl<true>([&principal,this,sra,mcc]() {
                                  principal.unscheduledFill(this->moduleLabel(), sra, mcc);},
                                resolveStatus);
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void
  ProducedProductHolder::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if(status() != defaultStatus()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << branchDescription().branchName() << "\n";
    }
    assert(edp.get() != nullptr);
    
    setProduct(std::move(edp));  // ProductHolder takes ownership
  }

  void
  InputProductHolder::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    setProduct(std::move(edp));
  }

  
  void
  DataManagingProductHolder::connectTo(ProductHolderBase const& iOther, Principal const*) {
    assert(false);
  }
  
  void
  DataManagingProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> prod) const {
    if(not prod) {return;}
    if(status() == defaultStatus()) {
      //resolveProduct has not been called or it failed
      putProduct(std::move(prod));
    } else {
      mergeProduct(std::move(prod));
    }
  }
  

  
  void
  DataManagingProductHolder::checkType(WrapperBase const& prod) const {
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
  DataManagingProductHolder::setProduct(std::unique_ptr<WrapperBase> edp) const {
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
  DataManagingProductHolder::productUnavailable_() const {
    auto presentStatus = status();
    if(presentStatus == ProductStatus::ProductSet) {
      return !(getProductData().wrapper()->isPresent());
    }
    return presentStatus != ProductStatus::ResolveNotRun;
  }
    
  bool
  DataManagingProductHolder::productResolved_() const {
    auto s = status();
    return (s != defaultStatus() ) or (s == ProductStatus::ProductDeleted);
  }

  
  // This routine returns true if the product was deleted early in order to save memory
  bool
  DataManagingProductHolder::productWasDeleted_() const {
    return status() == ProductStatus::ProductDeleted;
  }
  
  void DataManagingProductHolder::setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    productData_.setProvenance(provRetriever,ph,pid);
  }
  
  void DataManagingProductHolder::setProcessHistory_(ProcessHistory const& ph) {
    productData_.setProcessHistory(ph);
  }
  
  ProductProvenance const* DataManagingProductHolder::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }
  
  void DataManagingProductHolder::resetProductData_(bool deleteEarly) {
    if(theStatus_ == ProductStatus::ProductSet) {
      productData_.resetProductData();
    }
    if(deleteEarly) {
      theStatus_ = ProductStatus::ProductDeleted;
    } else {
      resetStatus();
    }
  }
  
  bool DataManagingProductHolder::singleProduct_() const {
    return true;
  }

  NoProcessProductHolder::
  NoProcessProductHolder(std::vector<ProductHolderIndex> const&  matchingHolders,
                         std::vector<bool> const& ambiguous) :
    matchingHolders_(matchingHolders),
    ambiguous_(ambiguous) {
    assert(ambiguous_.size() == matchingHolders_.size());
  }

  ProductData const* NoProcessProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
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
      if (matchingHolders_[k] != ProductHolderIndexInvalid) {
        ProductHolderBase const* productHolder = principal.getProductHolderByIndex(matchingHolders_[k]);
        ProductData const* pd =  productHolder->resolveProduct(resolveStatus, principal, skipCurrentProcess, sra, mcc);
        if(pd != nullptr) return pd;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void AliasProductHolder::setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    realProduct_.setProvenance(provRetriever,ph,pid);
  }

  void AliasProductHolder::setProcessHistory_(ProcessHistory const& ph) {
    realProduct_.setProcessHistory(ph);
  }

  ProductProvenance const* AliasProductHolder::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void AliasProductHolder::resetProductData_(bool deleteEarly) {
    realProduct_.resetProductData_(deleteEarly);
  }

  bool AliasProductHolder::singleProduct_() const {
    return true;
  }
  
  void AliasProductHolder::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
    << "AliasProductHolder::putProduct_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  
  void AliasProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
    << "AliasProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  

  
  void ParentProcessProductHolder::setProvenance_(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    provRetriever_ = provRetriever;
  }
  
  void ParentProcessProductHolder::setProcessHistory_(ProcessHistory const& ph) {
  }
  
  ProductProvenance const* ParentProcessProductHolder::productProvenancePtr_() const {
    return provRetriever_? provRetriever_->branchIDToProvenance(bd_->branchID()): nullptr;
  }
  
  void ParentProcessProductHolder::resetProductData_(bool deleteEarly) {
  }
  
  bool ParentProcessProductHolder::singleProduct_() const {
    return true;
  }

  void ParentProcessProductHolder::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
    << "ParentProcessProductHolder::putProduct_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  
  void ParentProcessProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
    << "ParentProcessProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }
  
  void NoProcessProductHolder::swap_(ProductHolderBase& rhs) {
    NoProcessProductHolder& other = dynamic_cast<NoProcessProductHolder&>(rhs);
    ambiguous_.swap(other.ambiguous_);
    matchingHolders_.swap(other.matchingHolders_);
  }

  void NoProcessProductHolder::setProvenance_(ProductProvenanceRetriever const* , ProcessHistory const& , ProductID const& ) {
  }

  void NoProcessProductHolder::setProcessHistory_(ProcessHistory const& ) {
  }

  ProductProvenance const* NoProcessProductHolder::productProvenancePtr_() const {
    return nullptr;
  }

  void NoProcessProductHolder::resetProductData_(bool) {
  }

  bool NoProcessProductHolder::singleProduct_() const {
    return false;
  }

  bool NoProcessProductHolder::unscheduledWasNotRun_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::unscheduledWasNotRun_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::productUnavailable_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::productUnavailable_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::productResolved_() const {
    throw Exception(errors::LogicError)
    << "NoProcessProductHolder::productResolved_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::productWasDeleted_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::productWasDeleted_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::putProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::putOrMergeProduct_(std::unique_ptr<WrapperBase> edp) not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  BranchDescription const& NoProcessProductHolder::branchDescription_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::branchDescription_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::resetBranchDescription_(std::shared_ptr<BranchDescription const>) {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::resetBranchDescription_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  Provenance const* NoProcessProductHolder::provenance_() const {
    throw Exception(errors::LogicError)
    << "NoProcessProductHolder::provenance_() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::connectTo(ProductHolderBase const&, Principal const*) {
    throw Exception(errors::LogicError)
    << "NoProcessProductHolder::connectTo() not implemented and should never be called.\n"
    << "Contact a Framework developer\n";
    
  }
}

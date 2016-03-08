/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "ProductHolders.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>
namespace {
  void
  mergeTheProduct(std::unique_ptr<edm::WrapperBase> iFrom,
                  edm::WrapperBase& ioTo,
                  edm::BranchDescription const& iBranchDescription) {
    if(ioTo.isMergeable()) {
      ioTo.mergeProduct(iFrom.get());
    } else if(ioTo.hasIsProductEqual()) {
      if(!ioTo.isProductEqual(iFrom.get())) {
        edm::LogError("RunLumiMerging")
        << "ProductHolder::mergeTheProduct\n"
        << "Two run/lumi products for the same run/lumi which should be equal are not\n"
        << "Using the first, ignoring the second\n"
        << "className = " << iBranchDescription.className() << "\n"
        << "moduleLabel = " << iBranchDescription.moduleLabel() << "\n"
        << "instance = " << iBranchDescription.productInstanceName() << "\n"
        << "process = " << iBranchDescription.processName() << "\n";
      }
    } else {
      edm::LogWarning("RunLumiMerging")
      << "ProductHolder::mergeTheProduct\n"
      << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
      << "Using the first, ignoring the second in merge\n"
      << "className = " << iBranchDescription.className() << "\n"
      << "moduleLabel = " << iBranchDescription.moduleLabel() << "\n"
      << "instance = " << iBranchDescription.productInstanceName() << "\n"
      << "process = " << iBranchDescription.processName() << "\n";
    }
  }

}

namespace edm {

  ProductData const*
  InputProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
                                      Principal const& principal,
                                      bool,
                                      SharedResourcesAcquirer* ,
                                      ModuleCallingContext const* mcc) const {
    if(productWasDeleted()) {
      throwProductDeletedException();
    }
    auto presentStatus = status();
    
    if(presentStatus == ProductStatus::ResolveNotRun) {
      //if readFromSource fails becuase of exception or not setting product
      // make sure the status goes to failed
      auto failedStatusSetter = [this](ProductStatus* presentStatus) {
        if(this->status() == ProductStatus::ResolveNotRun) {
          this->setFailedStatus();
        }
        *presentStatus = this->status();
      };
      std::unique_ptr<ProductStatus, decltype(failedStatusSetter)> failedStatusGuard(&presentStatus, failedStatusSetter);
      principal.readFromSource(*this, mcc);
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

  ProductData const*
  PuttableProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
                                          Principal const&,
                                          bool skipCurrentProcess,
                                          SharedResourcesAcquirer*,
                                          ModuleCallingContext const*) const {
    if (!skipCurrentProcess) {
      if(productWasDeleted()) {
        throwProductDeletedException();
      }
      if(status() == ProductStatus::ProductSet) {
        auto pd = &getProductData();
        if(pd->wrapper()->isPresent()) {
          resolveStatus = ProductFound;
          return pd;
        }
      }
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
      if(productWasDeleted()) {
        throwProductDeletedException();
      }
      auto presentStatus = status();

      if(presentStatus == ProductStatus::ResolveNotRun) {
        
        //if readFromSource fails becuase of exception or not setting product
        // make sure the status goes to failed
        auto failedStatusSetter = [this](ProductStatus* presentStatus) {
          if(this->status() == ProductStatus::ResolveNotRun) {
            this->setFailedStatus();
          }
          *presentStatus = this->status();
        };
        
        std::unique_ptr<ProductStatus, decltype(failedStatusSetter)> failedStatusGuard(&presentStatus, failedStatusSetter);

        principal.unscheduledFill(moduleLabel(), sra, mcc);
      }
      
      auto pd = &getProductData();
      if(presentStatus == ProductStatus::ProductSet && pd->wrapper()->isPresent()) {
        resolveStatus = ProductFound;
        return pd;
      }

    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  bool
  ProducedProductHolder::putOrMergeProduct_() const {
    return productUnavailable();
  }

  void
  ProducedProductHolder::mergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    assert(status() == ProductStatus::ProductSet);
    mergeTheProduct(std::move(edp), *(getProductData().unsafe_wrapper()),branchDescription());
  }

  void
  ProducedProductHolder::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if(status() != defaultStatus()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << branchDescription().branchName() << "\n";
    }
    assert(branchDescription().produced());
    assert(edp.get() != nullptr);
    assert(status() != ProductStatus::ProductSet);
    
    setProduct(std::move(edp));  // ProductHolder takes ownership
  }

  void
  InputProductHolder::mergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    assert(status() == ProductStatus::ProductSet);
    mergeTheProduct(std::move(edp), *(getProductData().unsafe_wrapper()),branchDescription());
  }

  bool
  InputProductHolder::putOrMergeProduct_() const {
    return(status() == defaultStatus());
  }

  void
  InputProductHolder::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    setProduct(std::move(edp));
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  InputProductHolder::productUnavailable_() const {
    // If there is a product, we know if it is real or a dummy.
    if(status() == ProductStatus::ProductSet) {
      return !(getProductData().wrapper()->isPresent());
    }
    return false;
  }
  
  void
  DataManagingProductHolder::connectTo(ProductHolderBase const& iOther, Principal const*) {
    assert(false);
  }
  

  void
  DataManagingProductHolder::setProduct(std::unique_ptr<WrapperBase> edp) const {
    if(edp) {
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
    // If unscheduled production, the product is potentially available.
    if(onDemandWasNotRun()) return false;
    // The product is available if and only if a product has been put.
    bool unavailable = !((status() == ProductStatus::ProductSet) && getProductData().wrapper()->isPresent());
    return unavailable;
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

  bool NoProcessProductHolder::onDemandWasNotRun_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::onDemandWasNotRun_() not implemented and should never be called.\n"
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

  void NoProcessProductHolder::mergeProduct_(std::unique_ptr<WrapperBase>) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::mergeProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::putOrMergeProduct_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::putOrMergeProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::checkType_(WrapperBase const&) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::checkType_() not implemented and should never be called.\n"
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
